"""exp74 Stage 0b — what the INPUT change alone does, at frozen parameters.

exp63's joint theta is held fixed. Every epoch is predicted twice: on the
official DiffMAH curves (the model as fitted) and on the pre-epoch curves of
`history.py` (the same four-number form, fitted only to the history before the
epoch, anchored there). The difference is what the model's knowledge of the
halo's future was doing, before any refit compensates for its removal.

Measured, fitting sample and mh-complete:
  1. G3 (reach): the fraction of the model's deposited stellar mass laid down
     before the first usable snapshot (1.18 Gyr) and before the official fit's
     2 Gyr cut -- how much of the input is extrapolation, per epoch;
  2. the profile change: median (pre-epoch / official - 1) and the median
     residual against the data at 2, 10, 52, 103 kpc, both curve sets;
  3. dy/dG, the residual's partial dependence on FUTURE growth at fixed catalog
     mass at 103 kpc (`selection.partial_growth`; exp71: model +0.125, truth
     +0.003) -- the leak itself;
  4. the halo-mass tilt of the 103 kpc residual at z = 2, fitting sample vs
     mh-complete (exp71: -0.122 vs +0.061) -- what C18 attributed to the
     selection acting through the leak;
  5. the half-mass radius, median model / truth, both curve sets;
  6. the standard battery's halo-mass-binned CoG figure for both.

Run after `history.py`:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u experiments/exp74_c19_history_leak/stage0_frozen.py [--smoke] [--tables-only]
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
          ROOT / "experiments/exp73_size_relative", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import selection as SEL                                  # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
import coordinate as C                                   # noqa: E402
import history as HI                                     # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE, THIN = "=" * 100, "-" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
FIT_NPZ = ROOT / "experiments/exp63_analytic_growth/outputs/stage2_fit_joint_kpc_free_sane.npz"
R_SHOW = (2.0, 10.25, 52.30, 103.45)
R_LEAK = 103.45
T_CUTS = (1.18, 2.0)


def reach(spec2, theta, curves):
    """Fraction of deposited stellar mass laid down before each T_CUT, per
    epoch, median over galaxies (G3)."""
    lt, w, include, _ = E.nodes(**M2.FULL_NODES)
    t = 10.0 ** lt
    frac = np.full((len(curves), len(EPOCHS), len(T_CUTS)), np.nan)
    for lo in range(0, len(curves), 300):
        cv = curves[lo:lo + 300]
        dm_c, dm_e, *_ = M2._deposits2(spec2, theta, cv, lt)
        wd = (dm_c + dm_e) * w[None, :]
        for k in EPOCHS:
            tot = (wd * include[k][None, :]).sum(1)
            for j, tc in enumerate(T_CUTS):
                frac[lo:lo + len(cv), k, j] = (wd * (include[k] & (t < tc))[None, :]).sum(1) / tot
    return frac


def tilt(y, x, mask):
    ok = mask & np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 30:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def main(smoke=False, tables_only=False, which="pre-epoch"):
    tag = ("_measured" if which == "measured" else "") + ("_smoke" if smoke else "")
    print(f"{RULE}\nexp74 Stage 0b — the input change at FROZEN theta (exp63 joint): {which} curves\n{RULE}\n")
    recs, data, mask_legacy, lmh_dm, _, _, _ = S0.build(smoke)
    fitmask, complete = S2F.fit_masks(data, mask_legacy, lmh_dm, "sane")
    curves = E.build_curves(recs, verbose=False)
    fz = np.load(FIT_NPZ, allow_pickle=True)
    spec2 = S2F.spec_from_fit(fz)
    theta = np.asarray(fz["theta_best"], float)
    hist_path = OUTDIR / f"history_curves{'_smoke' if smoke else ''}.npz"
    hist = dict(np.load(hist_path, allow_pickle=True))
    if which == "measured":
        import measured as MB
        hs0 = np.load(SEL.HS_NPZ, allow_pickle=True)
        meas, _ = MB.build_measured(recs, curves, hs0, hist, verbose=True)
        pre = {k: meas for k in EPOCHS}
        print(f"  {len(recs)} galaxies; theta = exp63 joint, frozen; MEASURED curves (one table per galaxy, "
              f"every epoch)")
    else:
        pre = {k: HI.curves_for_epoch(curves, hist, k) for k in EPOCHS}
        print(f"  {len(recs)} galaxies; theta = exp63 joint, frozen; pre-epoch curves from "
              f"{hist_path.name} (G1 {'OK' if hist['g1_ok'] else 'FAILED'}, G2 {'OK' if hist['g2_ok'] else 'FAILED'})")

    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    rows = np.array([h.row for h in recs])
    lmh_cat = SEL.sample_masses(hs)[rows]
    growth = lmh_cat[:, 0][:, None] - lmh_cat

    pred_off = M2.predict2(spec2, theta, curves, F.R_GRID)
    pred_pre = (M2.predict2(spec2, theta, pre[0], F.R_GRID) if which == "measured" else
                np.stack([M2.predict2(spec2, theta, pre[k], F.R_GRID, epochs=(k,))[:, 0, :] for k in EPOCHS], axis=1))
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for pr in (pred_off, pred_pre):
        good &= np.isfinite(pr).all(axis=(1, 2)) & (pr > 0).all(axis=(1, 2))
    fit_all = good & fitmask.all(1)
    print(f"  fitting sample {int(fit_all.sum())}; mh-complete per epoch "
          + "/".join(str(int((fit_all & complete[:, k]).sum())) for k in EPOCHS))

    # ---- 1. reach ---------------------------------------------------------- #
    print(f"\n{RULE}\n1. G3 REACH — fraction of the model's deposited mass laid down before "
          f"{T_CUTS[0]} Gyr (first usable snapshot)\n   and before {T_CUTS[1]} Gyr (the official "
          f"fit's own cut), official curves, median over the fitting sample\n{RULE}")
    fr = reach(spec2, theta, curves)
    print(f"  {'epoch':>6}{'< 1.18 Gyr':>14}{'< 2.0 Gyr':>14}")
    for k in EPOCHS:
        print(f"  {ANCHOR_Z[k]:>6}{100 * np.nanmedian(fr[fit_all, k, 0]):>13.1f}%{100 * np.nanmedian(fr[fit_all, k, 1]):>13.1f}%")

    # ---- 2. the profile change -------------------------------------------- #
    ir = [int(np.argmin(np.abs(F.R_GRID - r))) for r in R_SHOW]
    print(f"\n{RULE}\n2. THE PROFILE CHANGE — median (pre-epoch / official - 1) [%], then the median "
          f"residual (model - data)/data [%]\n   for each curve set; fitting sample | mh-complete\n{RULE}")
    print(f"  {'epoch':>6}{'':<12}" + "".join(f"{f'M(<{F.R_GRID[i]:.0f})':>22}" for i in ir))
    prof = {}
    for k in EPOCHS:
        m_c = fit_all & complete[:, k]
        ch = [(100 * np.nanmedian(pred_pre[fit_all, k, i] / pred_off[fit_all, k, i] - 1),
               100 * np.nanmedian(pred_pre[m_c, k, i] / pred_off[m_c, k, i] - 1)) for i in ir]
        print(f"  {ANCHOR_Z[k]:>6}{'change':<12}" + "".join(f"{a:>+10.1f} |{b:>+9.1f}" for a, b in ch))
        for lab, pr in (("official", pred_off), (which, pred_pre)):
            res = [(100 * np.nanmedian((pr[fit_all, k, i] - data[fit_all, k, i]) / data[fit_all, k, i]),
                    100 * np.nanmedian((pr[m_c, k, i] - data[m_c, k, i]) / data[m_c, k, i])) for i in ir]
            prof[(lab, k)] = res
            print(f"  {'':>6}{lab:<12}" + "".join(f"{a:>+10.1f} |{b:>+9.1f}" for a, b in res))

    # ---- 3. the leak ------------------------------------------------------- #
    c = int(np.argmin(np.abs(F.R_GRID - R_LEAK)))
    y_tru = np.log10(np.clip(data[:, :, c], 1.0, None))
    print(f"\n{RULE}\n3. THE LEAK — dy/dG at fixed CATALOG mass, M(<{R_LEAK:g} kpc) [dex per dex of future "
          f"growth]; exp71: model +0.125, truth +0.003 at z=2\n   (partial rho | slope); fitting sample\n{RULE}")
    print(f"  {'':<22}" + "".join(f"{f'z={z}':>18}" for z in ANCHOR_Z[1:]))
    leak = {}
    for lab, pr in (("truth", data), ("official model", pred_off), (f"{which} model", pred_pre),
                    ("residual, official", None), (f"residual, {which}", None)):
        cells = []
        for k in range(1, 5):
            if lab == "truth":
                y = y_tru[:, k]
            elif lab.startswith("residual"):
                src = pred_off if lab.endswith("official") else pred_pre
                y = np.log10(np.clip(src[:, k, c], 1.0, None)) - y_tru[:, k]
            else:
                y = np.log10(np.clip(pr[:, k, c], 1.0, None))
            rho, sl, n = SEL.partial_growth(y, lmh_cat[:, k], growth[:, k], fit_all & np.isfinite(lmh_cat[:, k]))
            leak[(lab, k)] = (rho, sl)
            cells.append(f"{rho:>+7.3f} |{sl:>+8.3f}")
        print(f"  {lab:<22}" + "".join(f"{s:>18}" for s in cells))

    # ---- 4. the z = 2 tilt ------------------------------------------------- #
    print(f"\n{RULE}\n4. THE HALO-MASS TILT of the {R_LEAK:g} kpc residual [dex per dex], fitting sample | "
          f"mh-complete; exp71 at z=2: -0.122 | +0.061\n   against the DiffMAH mass at z_k (the model's "
          f"binning variable) and the catalog mass\n{RULE}")
    print(f"  {'':<22}" + "".join(f"{f'z={z}':>20}" for z in ANCHOR_Z))
    tl = {}
    for lab, pr in (("official", pred_off), (which, pred_pre)):
        for xl, x in (("DiffMAH Mh", lmh_dm), ("catalog Mh", lmh_cat)):
            cells = []
            for k in EPOCHS:
                y = np.log10(np.clip(pr[:, k, c], 1.0, None)) - y_tru[:, k]
                a, b = tilt(y, x[:, k], fit_all), tilt(y, x[:, k], fit_all & complete[:, k])
                tl[(lab, xl, k)] = (a, b)
                cells.append(f"{a:>+8.3f} |{b:>+8.3f}")
            print(f"  {lab + ', ' + xl:<22}" + "".join(f"{s:>20}" for s in cells))

    # ---- 5. sizes ---------------------------------------------------------- #
    print(f"\n{RULE}\n5. THE HALF-MASS RADIUS — median model / truth - 1 [%], fitting sample\n{RULE}")
    r50_t = C.size_radius(data[fit_all], F.R_GRID, 0.5)
    print(f"  {'':<12}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z))
    sz = {}
    for lab, pr in (("official", pred_off), (which, pred_pre)):
        r50 = C.size_radius(pr[fit_all], F.R_GRID, 0.5)
        sz[lab] = [float(np.nanmedian(r50[:, k] / r50_t[:, k]) - 1) for k in EPOCHS]
        print(f"  {lab:<12}" + "".join(f"{100 * v:>+8.1f}%" for v in sz[lab]))

    # ---- 6. figures -------------------------------------------------------- #
    if not tables_only:
        (FIGDIR / "qa").mkdir(parents=True, exist_ok=True)
        logms = np.log10(np.clip(data[fit_all][:, 0, -1], 1.0, None))
        for lab, pr in (("official", pred_off), (which.replace("-", ""), pred_pre)):
            qa.evaluate(pr[fit_all], data[fit_all], F.R_GRID, ANCHOR_Z, name=f"exp74_frozen_{lab}{'_smoke' if smoke else ''}",
                        figdir=FIGDIR / "qa", figures=True, verbose=False,
                        bin_by=lmh_dm[fit_all][:, 0], bin_label=r"logM$_h$(z=0.4)",
                        bin_by_ms=logms, ms_label=r"logM$_*$ (total)")

    np.savez(OUTDIR / f"stage0_frozen{tag}.npz", reach=fr, fit_all=fit_all, complete=complete,
             prof_keys=np.array([f"{a}|{k}" for (a, k) in prof]), prof=np.array(list(prof.values())),
             leak_keys=np.array([f"{a}|{k}" for (a, k) in leak]), leak=np.array(list(leak.values())),
             tilt_keys=np.array([f"{a}|{b}|{k}" for (a, b, k) in tl]), tilt=np.array(list(tl.values())),
             size_official=sz["official"], size_pre=sz["pre-epoch"],
             pred_official=pred_off.astype(np.float32), pred_preepoch=pred_pre.astype(np.float32))
    print(f"\nwrote {OUTDIR / f'stage0_frozen{tag}.npz'}")


if __name__ == "__main__":
    a = sys.argv
    main(smoke="--smoke" in a, tables_only="--tables-only" in a,
         which=(a[a.index("--curves") + 1] if "--curves" in a else "pre-epoch"))
