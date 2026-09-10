"""exp80 — the evolution of R20, R50 and R80 with redshift, measured DIRECTLY
from the curves of growth (no deconvolution), on exp73's merged 0.673–148 kpc
grid so R20 is measured rather than extrapolated.

The user's request (2026-09-10): the statistical distributions of the three
fractional radii at each epoch, for the truth and the models, and the same
galaxies followed from z = 0.4 split into halo-mass and stellar-mass bins
defined AT z = 0.4 (their progenitors at the earlier epochs). Models: the
adopted baseline and Stage 0 C's frozen point (q_e = 0.127).

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp80_deposit_size_law/stage0_size_evolution.py [--smoke]
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

import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import size_terms as ST                                  # noqa: E402
import size_law as SL                                    # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S0T = _by_path("exp78_stage0_terms", ROOT / "experiments/exp78_size_aware_objective/stage0_terms.py")
RB = S0T.RB
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
FRACS = ((0.2, "R20"), (0.5, "R50"), (0.8, "R80"))
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"


def main(smoke=False):
    tag = "_smoke" if smoke else ""
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr, Rm, truth_m, n_inner, rows_m = S0T.build(smoke)
    spec_b, th_b = RB.adopted_baseline()
    rows = np.array(sorted(set.intersection(*[set(rows_m[k]) for k in EPOCHS])))   # usable at every epoch
    index_of = np.full(len(recs), -1); index_of[pr.all_rows] = np.arange(len(pr.all_rows))
    cv = [meas[i] for i in rows]
    d0 = np.load(OUTDIR / "stage0_cand_a_expand_free.npz", allow_pickle=True)
    th_f = th_b.copy()
    vals = dict(zip([str(n) for n in d0["free"]], np.asarray(d0["x_tuned"], float)))
    for n in ("log_f_e", "b_e"):
        th_f[spec2.index(n)] = vals[n]
    law_f = SL.with_law(q_e=float(vals["q_e"]))
    curves = {"truth": truth_m[rows],
              "baseline": SL.predict_law(spec2, th_b, SL.LAW_DEFAULT, cv, Rm, epochs=EPOCHS),
              "frozen q_e point": SL.predict_law(spec2, th_f, law_f, cv, Rm, epochs=EPOCHS)}
    sizes = {lab: {f: np.array([ST.fractional_radius(c[:, k], Rm, frac) for k in EPOCHS]).T
                   for frac, f in FRACS} for lab, c in curves.items()}     # (n, 5) each
    lmh0 = lmh_bins[rows, 0]
    lms0 = np.log10(truth_m[rows, 0, -1])
    bins = {}
    for nm, x in (("halo mass at z=0.4", lmh0), ("stellar mass at z=0.4", lms0)):
        e = np.quantile(x, [0, 1 / 3, 2 / 3, 1])
        bins[nm] = (e, [(x >= e[b]) & (x <= e[b + 1] + 1e-9) for b in range(3)])
    print(f"  {len(rows)} galaxies with a usable merged curve at every epoch; medians of R_f [kpc] (16-84%) per epoch")
    for f in ("R20", "R50", "R80"):
        for lab in curves:
            s = sizes[lab][f]
            print(f"    {f} {lab:<18}" + "".join(f"  z={ANCHOR_Z[k]}: {np.nanmedian(s[:, k]):5.1f} ({np.nanpercentile(s[:, k], 16):4.1f}-{np.nanpercentile(s[:, k], 84):5.1f})" for k in EPOCHS))
    for nm, (e, sel) in bins.items():
        print(f"  by {nm} (tercile edges {e[0]:.2f}/{e[1]:.2f}/{e[2]:.2f}/{e[3]:.2f}): median R50 [kpc] truth | baseline | frozen point")
        for b in range(3):
            print(f"    tercile {b}: " + "".join(
                f"  z={ANCHOR_Z[k]}: {np.nanmedian(sizes['truth']['R50'][sel[b], k]):5.1f} | {np.nanmedian(sizes['baseline']['R50'][sel[b], k]):5.1f} | {np.nanmedian(sizes['frozen q_e point']['R50'][sel[b], k]):5.1f}"
                for k in EPOCHS))
    np.savez(OUTDIR / f"stage0_size_evolution{tag}.npz", rows=rows, anchor_z=np.array(ANCHOR_Z), lmh0=lmh0, lms0=lms0,
             **{f"{lab.replace(' ', '_')}_{f}": sizes[lab][f] for lab in sizes for f in ("R20", "R50", "R80")})
    figure(sizes, bins, tag)


def figure(sizes, bins, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    from hongshao.qa import _tex
    set_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    zz = np.array(ANCHOR_Z)
    tc = ("#0072B2", "#009E73", "#D55E00")
    style = {"truth": dict(ls="-", lw=2.2), "baseline": dict(ls="--", lw=1.6), "frozen q_e point": dict(ls=":", lw=1.8)}
    fig, ax = plt.subplots(3, 3, figsize=(17, 13))
    # row 0: the whole-sample distributions per epoch, truth vs models, as 5-25-50-75-95 boxes
    for j, (frac, f) in enumerate(FRACS):
        a = ax[0, j]
        for i, (lab, st) in enumerate(style.items()):
            s = np.log10(sizes[lab][f])
            off = (i - 1) * 0.06
            for k in range(5):
                q = np.nanpercentile(s[:, k], [5, 16, 50, 84, 95])
                x = zz[k] + off
                a.plot([x, x], [q[0], q[4]], color=("k", tc[0], tc[2])[i], lw=0.8)
                a.plot([x, x], [q[1], q[3]], color=("k", tc[0], tc[2])[i], lw=4, alpha=0.6)
                a.plot(x, q[2], "o", color=("k", tc[0], tc[2])[i], ms=5)
            a.plot(zz + off, np.nanmedian(s, 0), color=("k", tc[0], tc[2])[i], label=lab, **st)
        a.set_xlabel("epoch z"); a.set_ylabel(_tex(f"log {f} [kpc]"))
        a.set_title(f"{f}: whole sample, median with 16-84 (thick) and 5-95 (thin)")
        a.legend(fontsize=8)
    # rows 1-2: the same galaxies split at z = 0.4, followed back
    for r, (nm, (e, sel)) in enumerate(bins.items(), start=1):
        for j, (frac, f) in enumerate(FRACS):
            a = ax[r, j]
            for b in range(3):
                for lab, st in style.items():
                    s = sizes[lab][f][sel[b]]
                    med = np.nanmedian(s, 0)
                    a.plot(zz, med, color=tc[b], **st,
                           label=(f"{['low', 'mid', 'high'][b]} {e[b]:.2f}-{e[b + 1]:.2f}" if lab == "truth" else
                                  (f"{lab}" if b == 0 else None)))
                    if lab == "truth":
                        a.fill_between(zz, np.nanpercentile(s, 16, 0), np.nanpercentile(s, 84, 0), color=tc[b], alpha=0.12)
            a.set_yscale("log"); a.set_xlabel("epoch z"); a.set_ylabel(_tex(f"{f} [kpc]"))
            a.set_title(f"{f} by {nm} (tercile medians; band = truth's 16-84)")
            a.legend(fontsize=7)
    fig.suptitle("the evolution of R20 / R50 / R80 measured from the curves of growth (merged 0.67-148 kpc grid): "
                 "truth (solid), baseline (dashed), the frozen q_e = 0.127 point (dotted); the same galaxies followed back from z = 0.4",
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, FIGDIR / f"exp80_size_evolution{tag}")
    print(f"  wrote {FIGDIR / f'exp80_size_evolution{tag}.png'}")


if __name__ == "__main__":
    main("--smoke" in sys.argv)
