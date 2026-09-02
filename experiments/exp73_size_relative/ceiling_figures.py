"""exp73 A2 closed — QA FIGURES for the five single-epoch R50 fits.

Two products:

1. The STANDARD battery (`hongshao.qa.evaluate`, figures on) for each
   `fit_re_e{k}.npz` — "fitted at one epoch, four predicted", the way exp63's
   per-epoch fits were shown. Names `exp73_R50_e{k}`; figures in `figures/qa/`.
   The halo-mass-binned CoG figure is the one to read; the stellar-mass-binned
   one carries the regression-to-the-mean offset by construction.

2. ONE summary figure, `figures/exp73_ceiling_own_epoch.png`: at each epoch
   (columns), in each halo-mass tercile (rows), the median (model - data)/data
   on the merged 0.673-148 kpc grid for the four models that matter there —
   the R50 per-epoch fit (its OWN epoch), exp63's per-epoch fit, the R50 joint
   refit and exp63's joint fit. This is the ceiling question in one picture:
   how much better can an epoch be served on its own, and where.

Run after `ceiling_re.py`:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u experiments/exp73_size_relative/ceiling_figures.py [--smoke] [--summary-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

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
import fit_re as FR                                      # noqa: E402
from ceiling_re import load_models                       # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE_KEYS = ("kpc:M(<10)", "kpc:M(<100)")


def summary_figure(cog_m, truth_m, Rm, lmh0, tag):
    """3 terciles x 5 epochs of median residual curves on the merged grid."""
    edges = np.quantile(lmh0, [0, 1 / 3, 2 / 3, 1])
    styles = {"R50 e": ("-", 2.0, "C3"), "exp63 e": ("--", 1.6, "C0"),
              "R50 joint": ("-", 1.2, "C1"), "exp63 joint": (":", 1.6, "0.3")}
    fig, axes = plt.subplots(3, 5, figsize=(17, 8.5), sharex=True, sharey=True)
    for b in range(3):
        m = (lmh0 >= edges[b]) & (lmh0 <= edges[b + 1] + 1e-9)
        for k in EPOCHS:
            ax = axes[b, k]
            for lab, (ls, lw, col) in styles.items():
                n = f"{lab}{k}" if lab.endswith(" e") else lab
                if n not in cog_m:
                    continue
                r = (cog_m[n][m, k, :] - truth_m[m, k, :]) / truth_m[m, k, :]
                ax.plot(Rm, 100 * np.nanmedian(r, axis=0), ls, lw=lw, c=col,
                        label=(f"R50 fit at THIS epoch" if lab == "R50 e" else
                               f"exp63 fit at this epoch (fixed kpc)" if lab == "exp63 e" else
                               f"R50 joint refit" if lab == "R50 joint" else "exp63 joint"))
            ax.axhline(0, c="0.6", lw=0.8)
            for y in (-10, 10):
                ax.axhline(y, c="0.85", lw=0.7, ls=":")
            ax.axvline(2.0, c="0.85", lw=0.7)
            ax.set(xscale="log", ylim=(-25, 25))
            if b == 0:
                ax.set_title(qa._tex(f"z={ANCHOR_Z[k]}"), fontsize=11)
            if b == 2:
                ax.set_xlabel("R [kpc]")
            if k == 0:
                ax.set_ylabel(qa._tex(f"logM$_h$(z=0.4) {edges[b]:.2f}-{edges[b + 1]:.2f}\n"
                                      f"median (model$-$data)/data [{qa._pct()}]"), fontsize=9)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(qa._tex("exp73 — the per-epoch ceiling under the R50 objective, at each "
                         "epoch's OWN fit (merged grid, 0.673-148 kpc; 2 kpc marked)"),
                 fontsize=12)
    fig.tight_layout()
    print("wrote", qa.save_fig(fig, FIGDIR / f"exp73_ceiling_own_epoch{tag}")[0])


def main(smoke=False, summary_only=False):
    tag = "_smoke" if smoke else ""
    print(f"{RULE}\nexp73 A2 closed — QA figures for the five single-epoch R50 fits\n{RULE}\n")
    (spec2, th_inc, th_nested, fz63, curves, truth_m, Rm, lmh0,
     rows, data, lmh, good) = FR.build(smoke)
    models, _ = load_models(spec2, th_nested, fz63, tag)
    names = [n for n in models if n != "nested incumbent"]
    cog_m = {n: M2.predict2(sp, th, curves, Rm, nodes=M2.FULL_NODES) for n, (sp, th) in models.items() if n in names}
    cog_k = {n: M2.predict2(sp, th, curves, F.R_GRID, nodes=M2.FULL_NODES) for n, (sp, th) in models.items() if n in names}
    ok = np.isfinite(truth_m).all(axis=(1, 2))
    for c in list(cog_m.values()) + list(cog_k.values()):
        ok &= np.isfinite(c).all(axis=(1, 2)) & (c > 0).all(axis=(1, 2))
    print(f"  {int(ok.sum())} galaxies")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    summary_figure({n: c[ok] for n, c in cog_m.items()}, truth_m[ok], Rm, lmh0[ok], tag)
    if summary_only:
        return

    d_k = data[good][ok]
    logms = np.log10(np.clip(d_k[:, 0, -1], 1.0, None))
    (FIGDIR / "qa").mkdir(parents=True, exist_ok=True)
    out = {}
    for k in EPOCHS:
        n = f"R50 e{k}"
        if n not in cog_k:
            continue
        print(f"\n  --- {n}: fitted at z={ANCHOR_Z[k]} alone, the other four predicted ---")
        out[n] = qa.evaluate(cog_k[n][ok], d_k, F.R_GRID, ANCHOR_Z,
                             name=f"exp73_R50_e{k}{tag}", figdir=FIGDIR / "qa",
                             figures=True, verbose=False,
                             bin_by=lmh0[ok], bin_label=r"logM$_h$(z=0.4)",
                             bin_by_ms=logms, ms_label=r"logM$_*$ (total)",
                             halo_mass_epochs=lmh[good][ok])
        qa.print_size_gate(out[n]["size_gate_ms"], ANCHOR_Z, "STELLAR mass")
    for key in TABLE_KEYS:
        print(f"\n  {key} — median relative bias, fitting sample; the fitted epoch in [brackets]")
        print(f"  {'model':<12}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for k in EPOCHS:
            n = f"R50 e{k}"
            if n not in out:
                continue
            t, mm = out[n]["truth"][key], out[n]["model"][key]
            cells = []
            for j in range(5):
                v = 100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j]))
                cells.append(f"[{v:+.1f}%]".rjust(10) if j == k else f"{v:+.1f}%".rjust(10))
            print(f"  {n:<12}" + "".join(cells))
    print(f"\n  figures -> {FIGDIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, summary_only="--summary-only" in sys.argv)
