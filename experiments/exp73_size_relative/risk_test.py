"""exp73 Block A2 — THE RISK TEST. Is the coordinate the limit, or the shared law?

This runs before any weighting sweep and before any fit, because it can make
them pointless. It asks the two questions the exp73 plan named as the risk, and
it needs no new fitting: every model it scores already exists.

THE SETUP. exp63 fitted its twelve-parameter model two ways on the same sample:
once with ONE theta for all five epochs (`stage2_fit_joint_kpc_free_sane`) and
once with a separate theta per epoch (`stage2_fit_frozen_binned_kpc_sane_e0..4`,
which freeze the four TIME exponents `a_z`, `a_Mz`, `b_c`, `b_e` because a single
epoch cannot constrain them). By exp63's own gate the per-epoch fits are about
2.3 to 2.7 times better at every epoch. That gap is the price of sharing one law
across cosmic time, and the whole question is what causes it.

QUESTION 1 — DOES THE GAP NARROW UNDER A SIZE-RELATIVE COORDINATE?

    gap = loss(one shared theta) / loss(that epoch's own theta)

measured under two objectives that differ in ONE respect: whether the shells are
at fixed kpc or at fixed multiples of the galaxy's own R50. Everything else --
the shell count, the mass-weighted residual, the sample, the models -- is held
identical, and the fixed-kpc edges are set to the median R50 shell edges at
z = 0.4 so the two span the same part of the galaxy at the anchor epoch.

  If the gap narrows materially, part of what looked like the cost of sharing a
  law was really the cost of scoring five different-sized galaxies on one fixed
  ruler, and the coordinate is worth pursuing.
  If it does not, the shared law is the limit and no objective will fix it.

QUESTION 2 — DO THE PER-EPOCH OPTIMA LIE ON A SMOOTH CURVE IN TIME?

If they do, a shared law with a richer time parametrisation could in principle
reach them and the gap is a parametrisation problem. If they scatter, the
per-epoch fits are chasing something no smooth law can follow.

This is sharper than it sounds, because of what the per-epoch fits froze.
**`b_c` is frozen at 0**, so the compact deposit size is `s_c = 10^log_f_c` with
no time dependence at all -- which makes the five fitted `log_f_c` values a
NON-PARAMETRIC measurement of how the compact size runs with redshift. The same
holds for `log_f_e` relative to its frozen `b_e`. So asking whether one power of
(1 + z) is enough becomes: is `log_f_c` linear in `log10(1 + z)`?

**z = 2 is reported but EXCLUDED from the trend fits.** exp63 Stage 5i found the
per-epoch sequence breaks there and read it as "partly a mass-selection law";
exp71 then showed the z = 2 halo-mass dependence IS the z = 0.4 progenitor
selection. Fitting a time law through a point that is largely selection would be
fitting the selection.

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u \\
        experiments/exp73_size_relative/risk_test.py [--smoke]
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
from hongshao.profile_data import load_profiles          # noqa: E402

OUTDIR = HERE / "outputs"
RULE, THIN = "=" * 100, "-" * 100
ANCHOR_Z = E.ANCHOR_Z
EPOCHS = (0, 1, 2, 3, 4)
E63 = ROOT / "experiments/exp63_analytic_growth/outputs"
POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
#: the epochs the time-law trend is fitted on. z = 2 is reported and excluded --
#: see the module docstring.
TREND_EPOCHS = (0, 1, 2, 3)


def binned_loss(grid, resid, lmh0, n_bin=3):
    """exp63's gate, translated to any shell coordinate.

    The programme judges by the MEDIAN profile in halo-mass bins, not by
    per-galaxy accuracy, and the two are not interchangeable: a per-galaxy loss
    trades the bin-median offset away (memory `loss-is-blind-to-the-binned-gate`,
    and exp63 added `--objective binned` for exactly this reason). Measured here:
    a per-galaxy rms puts the joint fit and the per-epoch fits within 2 per cent
    of each other, while exp63's own gate separates them by 2.3-2.7x. So a risk
    test that used a per-galaxy metric would conclude there is no gap to close.

    Returns (n_epoch,) rms over shells and bins of the per-bin median residual.
    """
    e = np.percentile(lmh0, np.linspace(0, 100, n_bin + 1))
    out = np.zeros(resid.shape[1])
    for k in range(resid.shape[1]):
        acc = []
        for b in range(n_bin):
            m = (lmh0 >= e[b]) & (lmh0 <= e[b + 1] + 1e-9)
            if m.sum() < 10:
                continue
            with np.errstate(invalid="ignore"):
                acc.append(np.nanmedian(resid[m, k, :], axis=0))
        out[k] = np.sqrt(np.nanmean(np.array(acc) ** 2)) if acc else np.nan
    return out


def gap_table(losses):
    """losses[model][k] -> the ratio joint / that-epoch's-own-theta."""
    return np.array([losses["joint"][k] / losses[f"e{k}"][k] for k in EPOCHS])


def trend(par, z, epochs=TREND_EPOCHS):
    """Fit `par` against log10(1+z) on `epochs`; return (slope, rms residual,
    full range, rms of a quadratic)."""
    x = np.log10(1.0 + np.asarray(z)[list(epochs)])
    y = np.asarray(par)[list(epochs)]
    if not np.isfinite(y).all():
        return np.nan, np.nan, np.nan, np.nan
    c1 = np.polyfit(x, y, 1)
    r1 = float(np.sqrt(np.mean((y - np.polyval(c1, x)) ** 2)))
    c2 = np.polyfit(x, y, 2)
    r2 = float(np.sqrt(np.mean((y - np.polyval(c2, x)) ** 2)))
    return float(c1[0]), r1, float(y.max() - y.min()), r2


def main(smoke=False):
    print(f"{RULE}\nexp73 Block A2 — THE RISK TEST: the coordinate, or the "
          f"shared law?\n{RULE}\n")
    (recs, data, lmh, curves, sel, spec2, th_inc, th_nested, fz63,
     keep, sig_cog, sig_ann, ann_ok) = G.build(smoke)
    good = keep.all(1)

    # ---- the models: one joint theta, and each epoch's own ---------------- #
    thetas = {"joint": np.asarray(fz63["theta_best"], float)}
    specs = {"joint": spec2}
    for k in EPOCHS:
        fz = np.load(E63 / f"stage2_fit_frozen_binned_kpc_sane_e{k}.npz", allow_pickle=True)
        thetas[f"e{k}"] = np.asarray(fz["theta_best"], float)
        specs[f"e{k}"] = S2F.spec_from_fit(fz)
    names = list(thetas)
    print(f"  one joint theta and five per-epoch thetas, all from exp63, none "
          f"refitted here")

    # ---- the two coordinates, differing in one respect only --------------- #
    pop = np.load(POP, allow_pickle=True)
    # `pop['index']` maps a POPULATION row to its row in the profile table, and
    # `sel` maps our records to population rows -- so the profile rows for these
    # galaxies are idx[sel], and getting that composition wrong silently
    # mismatches galaxies rather than raising
    idx = np.asarray(pop["index"], int)[sel]
    d = load_profiles()
    Rm, truth_all = C.extended_cog(data, d["cog_from_density_shared"][idx],
                                   F.R_GRID, d["shared_grid_kpc"])
    # the model is analytic, so evaluate it ON THE MERGED RADII rather than
    # interpolating a 24-radius prediction onto a 31-radius grid -- an
    # interpolated model would carry its own inner error into the comparison
    cogs = {n: M2.predict2(specs[n], thetas[n], curves, Rm, nodes=M2.FULL_NODES)
            for n in names}
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    good &= np.isfinite(truth_all).all(axis=(1, 2)) & (truth_all > 0).all(axis=(1, 2))
    truth_m = truth_all[good]
    print(f"  {int(good.sum())} galaxies on a merged {len(Rm)}-radius grid, "
          f"{Rm[0]:.3f} to {Rm[-1]:.2f} kpc")
    g_re = C.SizeGrid(truth_m, Rm)                        # R50 shells

    class FixedGrid(C.SizeGrid):
        """The same machinery with the shells pinned to fixed kpc."""
        def __init__(self, truth, R, edges):
            self.R = np.asarray(R, float)
            self.edges = tuple(edges)
            t = np.asarray(truth, float)
            self.r50 = np.ones(t.shape[:2])
            self.radii = np.broadcast_to(np.asarray(edges, float),
                                         t.shape[:2] + (len(edges),)).copy()
            self.total = t[..., -1]
            self.truth_shells = self.shells(t)
            self.total_re = self.cum(t)[..., -1]
            inside = ((self.radii[..., :-1] >= self.R[0])
                      & (self.radii[..., 1:] <= self.R[-1]))
            self.covered = inside & (self.truth_shells > 0)


    # ---- 1. the gap: a 2x2 of coordinate against residual form ----------- #
    #
    # The shell COUNT is matched to exp63's 24 radii, because a coarse objective
    # cannot answer this: a first attempt with 4 shells and a mass-weighted
    # residual put every model within 5 per cent of every other, while exp63's
    # own gate separates them by 2.3-2.7x. A metric that cannot see the gap
    # cannot report whether the coordinate closes it.
    n_edge = 21
    re_edges = tuple(np.geomspace(0.5, 6.0, n_edge))
    r50_z0 = float(np.nanmedian(g_re.r50[:, 0]))
    kpc_edges = tuple(r50_z0 * np.array(re_edges))
    grids = {"R50 shells": C.SizeGrid(truth_m, Rm, edges=re_edges),
             "fixed kpc": FixedGrid(truth_m, Rm, kpc_edges)}
    print(f"\n{RULE}\n1. DOES THE GAP NARROW? — a 2x2. Rows are the coordinate "
          f"(shells at fixed multiples\n   of the galaxy's OWN R50, or at fixed "
          f"kpc); columns are the residual form\n   (relative to the shell's own "
          f"mass, the programme's incumbent, or to the galaxy's\n   TOTAL mass). "
          f"{n_edge - 1} shells in every case, spanning 0.5-6 R50, with the fixed-kpc "
          f"edges\n   set to {r50_z0:.2f} kpc x those multiples so the two agree "
          f"at z = 0.4.\n{RULE}")
    print(f"\n   gap = loss(ONE SHARED theta) / loss(THAT EPOCH'S OWN theta). "
          f"1.00 = sharing is free.")
    res, gaps = {}, {}
    for cname, grid in grids.items():
        w = C.weights_uniform(grid)
        for kind in ("rel", "mass"):
            for how in ("per-galaxy", "binned"):
                tag = f"{cname:<11} | {kind:<4} | {how}"
                losses = {}
                for n in names:
                    r = grid.residual(cogs[n][good], kind)
                    if how == "per-galaxy":
                        per = C.apply_weights(r, w)
                        losses[n] = np.array([np.nanmean(per[:, k]) for k in EPOCHS])
                    else:
                        losses[n] = binned_loss(grid, r, lmh[good][:, 0])
                res[tag] = losses
                gaps[tag] = gap_table(losses)
    print(f"\n  {'coordinate | residual | how':<34}"
          + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z)
          + f"{'z<=1.5':>10}")
    for tag in gaps:
        print(f"  {tag:<34}" + "".join(f"{v:>9.2f}" for v in gaps[tag])
              + f"{np.mean(gaps[tag][:4]):>10.2f}")

    print(f"\n{THIN}\n  THE COMPARISON THAT ANSWERS THE QUESTION — same residual "
          f"form, the coordinate\n  swapped. z = 2 excluded: exp71/C18 showed its "
          f"halo-mass dependence is the z = 0.4\n  progenitor selection, so a "
          f"gap there is partly selection, not law.\n{THIN}")
    print(f"\n  THE CAVEAT THAT LIMITS THIS, and it is not small. The five "
          f"per-epoch fits were\n  optimised under exp63's FIXED-KPC binned "
          f"objective, so they are not the per-epoch\n  optimum under an R50 "
          f"one — and it shows: the R50 binned gap falls BELOW 1 at\n  "
          f"z >= 1.0, meaning the shared theta beats the epoch-specialised "
          f"models there. Part of\n  the apparent narrowing is therefore a "
          f"mis-specialised comparator, not a real\n  reduction in the cost of "
          f"sharing. Settling this needs five single-epoch refits\n  under the "
          f"R50 objective — a fit, and the only way to a true ceiling.\n")
    for how in ("binned", "per-galaxy"):
        for kind, label in (("rel", "relative residual (the incumbent form)"),
                            ("mass", "mass-weighted residual")):
            a_ = np.mean(gaps[f"{'fixed kpc':<11} | {kind:<4} | {how}"][:4])
            b_ = np.mean(gaps[f"{'R50 shells':<11} | {kind:<4} | {how}"][:4])
            print(f"  {how:<11} {label:<40} fixed kpc {a_:.2f}  ->  R50 {b_:.2f}"
                  f"   ({100 * (1 - b_ / a_):+.1f}%)")

    # ---- 2. do the per-epoch optima lie on a smooth curve? ---------------- #
    print(f"\n{RULE}\n2. DO THE PER-EPOCH OPTIMA LIE ON A SMOOTH CURVE IN TIME? — "
          f"each free parameter\n   against log10(1+z), fitted on z <= 1.5 and "
          f"with z = 2 shown but EXCLUDED.\n   `resid/range` is the scatter about "
          f"a straight line as a fraction of the span the\n   parameter covers: "
          f"small means one power of (1+z) already describes it.\n{RULE}")
    fz0 = np.load(E63 / "stage2_fit_frozen_binned_kpc_sane_e0.npz", allow_pickle=True)
    pnames = [str(x) for x in fz0["theta_names"]]
    frozen = {str(a): float(b) for a, b in zip(fz0["frozen_names"], fz0["frozen_values"])}
    free = [p for p in pnames if p not in frozen]
    print(f"\n  the per-epoch fits FROZE {list(frozen)} (a single epoch cannot "
          f"constrain a time exponent),\n  so the {len(free)} free parameters "
          f"below carry the whole time dependence "
          f"NON-PARAMETRICALLY.\n  `b_c` is frozen at 0, so `log_f_c` IS the "
          f"compact size in log10 kpc at that epoch.")
    tab = np.full((len(free), 5), np.nan)
    for i, p in enumerate(free):
        j = pnames.index(p)
        tab[i] = [thetas[f"e{k}"][j] for k in EPOCHS]
    print(f"\n  {'parameter':<10}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z)
          + f"{'slope':>9}{'resid':>8}{'range':>8}{'resid/range':>13}")
    out = {}
    for i, p in enumerate(free):
        sl, r1, rng, r2 = trend(tab[i], ANCHOR_Z)
        out[p] = (sl, r1, rng, r2)
        # no flag: an earlier version thresholded resid/range at 0.05 and fired
        # on all eight parameters, which is not a discriminator
        flag = ""
        print(f"  {p:<10}" + "".join(f"{v:>9.3f}" for v in tab[i])
              + f"{sl:>9.3f}{r1:>8.3f}{rng:>8.3f}"
              + f"{(r1 / rng if rng else np.nan):>13.3f}" + flag)
    print(f"\n  A quadratic was tried and is NOT reported: three parameters "
          f"through four points\n  removes residual by construction, so it "
          f"cannot discriminate. What the table says is\n  read from `resid` "
          f"directly — for the two SIZE parameters a straight line in\n  "
          f"log10(1+z) leaves {out['log_f_c'][1]:.3f} dex on the compact size "
          f"and {out['log_f_e'][1]:.3f} dex on the extended one,\n  i.e. "
          f"{100 * (10 ** out['log_f_c'][1] - 1):.0f}% and "
          f"{100 * (10 ** out['log_f_e'][1] - 1):.0f}% in size. One power of "
          f"(1+z) already describes the size evolution\n  below z = 2 to a few "
          f"per cent, so the user's D3 lever looks SMALL.")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDIR / ("risk_test_smoke.npz" if smoke else "risk_test.npz"),
             anchor_z=np.array(ANCHOR_Z), free=np.array(free), per_epoch=tab,
             kpc_edges=np.array(kpc_edges),
             **{f"gap_{t.replace(' ', '_')}": gaps[t] for t in gaps},
             **{f"loss_{t.replace(' ', '_')}_{n}": res[t][n] for t in res for n in names})
    print(f"\nwrote {OUTDIR / 'risk_test.npz'}")
    return gaps, out


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
