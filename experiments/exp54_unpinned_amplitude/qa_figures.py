"""exp54 — QA figures for the top-ranked deposition models.

Three figures per model, all with the truth SOLID and the model DASHED, one
colour per observation epoch:

  `qa_cog_<label>`     the AVERAGE curve of growth, in three views. Col 1: all
                       2397 galaxies. Col 2: each epoch's OWN fair sample --
                       a different galaxy set at each epoch, so its
                       epoch-to-epoch differences are NOT evolution. Col 3: one
                       FIXED set (fair at z=2) followed everywhere, which is
                       both completeness-controlled and a real evolutionary
                       sequence. Cols 1 and 3 may be read as evolution; col 2
                       may not, and mistaking it for one overstates the inner
                       growth by a factor of four.

  `qa_cogmass_<label>` the same curves split into terciles of HALO mass. Binned
                       by a model INPUT, never by the truth stellar mass:
                       selecting on the noisy quantity tilts the residual by
                       (slope - 1) dex per dex for any conditional-mean model,
                       and that artifact is strongest at small radii, exactly
                       where the model's known defect lives.

  `qa_resid_<label>`   the residual alone, full against fair, at every epoch,
                       with the population scatter as a band.

TERMS USED IN THE AXES, none of which should be read as obvious:

  **curve of growth (CoG)**  `M*(<R)`, the stellar mass enclosed within a
    sphere of radius R. It is cumulative, so it can only rise with R.

  **residual**  `(model - truth)/truth` at fixed R, in per cent. Negative means
    the model puts LESS stellar mass inside R than the simulation does.

  **amplitude-pinned residual**  the same after rescaling each model curve so
    its total at 148 kpc matches that galaxy's true total. It removes every
    error in HOW MUCH mass the model made and leaves only error in WHERE the
    mass was put. Plotting both separates a "wrong total" failure from a
    "wrong distribution" failure, which look identical in the raw residual.

  **fair sample**  the galaxies at each epoch whose halo is massive enough that
    being a progenitor of a z=0.4-selected galaxy is no longer a strong extra
    condition (`selection.py`). At z=0.4 that is all 2397; at z=2 it is 840.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
     experiments/exp54_unpinned_amplitude/qa_figures.py [--label L]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import halo as H                                         # noqa: E402
import stage33 as S33                                    # noqa: E402
from hongshao.plotting import save_fig, set_style        # noqa: E402
from hongshao.qa import _pct, _tercile_curves, _tex, _zcolors   # noqa: E402

POP = ROOT / "experiments/exp32_full_population/outputs/population.npz"
SEL = HERE / "outputs" / "selection.npz"
REFIT = HERE / "outputs" / "stage33_refit.npz"
PEREP = HERE / "outputs" / "stage33_perepoch.npz"
FIGDIR = HERE / "figures"

Z = (0.4, 0.7, 1.0, 1.5, 2.0)
R = F.R_GRID


def _predict(label, theta, recs, data):
    pr = F.Problem(S33.spec_from_label(label), recs, data, epochs=(0, 1, 2, 3, 4))
    return pr.predict(theta)


def cog_figure(label, model, truth, mask, name):
    """Average curve of growth, three views that must not be confused.

    THE TRAP THIS FIGURE IS BUILT TO AVOID. Column 2 uses a DIFFERENT set of
    galaxies at each epoch -- 2397 at z=0.4 falling to 840 at z=2 -- because the
    fair-sample cut removes the halo masses where our sample is badly
    incomplete, and which masses those are depends on the epoch. Reading its
    epoch-to-epoch differences as EVOLUTION is wrong: most of what changes is
    the sample. Measured at 2.8 kpc, the apparent inner rise from z=0.4 to z=2
    is +0.247 dex as plotted, of which only +0.062 dex survives when the same
    galaxies are followed -- **75% of it is the changing sample.**

    Column 3 fixes that: one FIXED set (the galaxies fair at z=2) followed at
    every epoch, so it is both completeness-controlled and a genuine
    evolutionary sequence. Columns 1 and 3 may be read as evolution; column 2
    may not.
    """
    import matplotlib.pyplot as plt
    set_style()
    cols = _zcolors(len(Z))
    fixed = mask[:, 4]                      # fair at z=2, followed everywhere
    fixed_col = np.repeat(fixed[:, None], len(Z), axis=1)
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 7.2), sharex=True,
                           height_ratios=[2, 1])
    rmax = 0.0
    for c, (sel, ttl) in enumerate((
            (np.ones_like(mask), "all 2397 galaxies\n(same set every epoch = evolution)"),
            (mask, "each epoch's own fair sample\n(DIFFERENT set each epoch - NOT evolution)"),
            (fixed_col, f"fixed set: the {int(fixed.sum())} fair at z=2\n"
                        f"(same set every epoch = evolution)"))):
        for k in range(len(Z)):
            g = sel[:, k] & np.isfinite(model[:, k, :]).all(axis=1)
            if g.sum() < 10:
                continue
            med_d, med_m, rel, rel_pin = _tercile_curves(model[g, k], truth[g, k])
            rmax = max(rmax, np.nanmax(np.abs(rel)))
            lab = f"z = {Z[k]}  (n={int(g.sum())})"
            ax[0, c].plot(R, med_d, "-", c=cols[k], lw=1.9, label=lab)
            ax[0, c].plot(R, med_m, "--", c=cols[k], lw=1.4)
            ax[1, c].plot(R, rel, "-", c=cols[k], lw=1.5)
            ax[1, c].plot(R, rel_pin, ":", c=cols[k], lw=1.3)
        ax[0, c].set_title(_tex(ttl), fontsize=10)
        ax[0, c].set_xscale("log")
        ax[1, c].set_xscale("log")
        ax[1, c].axhline(0, c="0.6", lw=0.8)
        for y in (-10, 10):
            ax[1, c].axhline(y, c="0.85", lw=0.7, ls=":")
        ax[1, c].set_xlabel(_tex("R [kpc]"))
    lim = float(np.clip(1.3 * rmax, 8.0, 60.0))
    for c in (0, 1, 2):
        ax[1, c].set_ylim(-lim, lim)
    ax[0, 0].set_ylabel(_tex("median log10 M*(<R)  [Msun]"))
    ax[1, 0].set_ylabel(_tex(f"median (model - truth)/truth  [{_pct()}]"))
    ax[0, 0].legend(loc="lower right", fontsize=8)
    ax[1, 0].plot([], [], "-", c="0.3", label=_tex("raw"))
    ax[1, 0].plot([], [], ":", c="0.3", label=_tex("amplitude-pinned (shape only)"))
    ax[1, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle(_tex(f"exp54 {label} — average curve of growth "
                      f"(truth solid, model dashed)"), fontsize=12)
    fig.text(0.5, 0.005, _tex(
        "All three columns show the SAME model; only the galaxies averaged "
        "differ. COLUMN 2 USES A DIFFERENT SET AT EACH EPOCH (2397 at z=0.4 "
        "falling to 840 at z=2), so its epoch-to-epoch differences are NOT "
        "evolution: at 2.8 kpc the apparent rise from z=0.4 to z=2 is +0.247 "
        "dex, of which only +0.062 dex survives when the same galaxies are "
        "followed. Columns 1 and 3 hold the galaxy set fixed and may be read "
        "as evolution. Dotted residual = each model curve rescaled to its "
        "galaxy's true total, so it shows only WHERE the mass was put, not "
        "HOW MUCH was made."),
        ha="center", va="bottom", fontsize=7.0, style="italic", color="0.35",
        wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    print("wrote", save_fig(fig, FIGDIR / f"qa_cog_{name}")[0], flush=True)


def cogmass_figure(label, model, truth, lmh, name):
    """The same curves, split into terciles of HALO mass (a model input)."""
    import matplotlib.pyplot as plt
    set_style()
    cols = _zcolors(len(Z))
    edges = np.quantile(lmh, [0, 1 / 3, 2 / 3, 1])
    fig, ax = plt.subplots(2, 3, figsize=(15.0, 7.2), sharex=True,
                           height_ratios=[2, 1])
    rmax = 0.0
    for b in range(3):
        sel = (lmh >= edges[b]) & (lmh <= edges[b + 1] + 1e-9)
        for k in range(len(Z)):
            g = sel & np.isfinite(model[:, k, :]).all(axis=1)
            if g.sum() < 10:
                continue
            med_d, med_m, rel, rel_pin = _tercile_curves(model[g, k], truth[g, k])
            rmax = max(rmax, np.nanmax(np.abs(rel)))
            ax[0, b].plot(R, med_d, "-", c=cols[k], lw=1.9,
                          label=f"z = {Z[k]}" if b == 0 else None)
            ax[0, b].plot(R, med_m, "--", c=cols[k], lw=1.4)
            ax[1, b].plot(R, rel, "-", c=cols[k], lw=1.5)
            ax[1, b].plot(R, rel_pin, ":", c=cols[k], lw=1.3)
        ax[0, b].set_title(_tex(f"log10 Mh(z=0.4) {edges[b]:.2f}-"
                                f"{edges[b+1]:.2f}   (n={int(sel.sum())})"),
                           fontsize=10)
        ax[0, b].set_xscale("log")
        ax[1, b].set_xscale("log")
        ax[1, b].axhline(0, c="0.6", lw=0.8)
        ax[1, b].set_xlabel(_tex("R [kpc]"))
    lim = float(np.clip(1.3 * rmax, 8.0, 60.0))
    for b in range(3):
        ax[1, b].set_ylim(-lim, lim)
    ax[0, 0].set_ylabel(_tex("median log10 M*(<R)  [Msun]"))
    ax[1, 0].set_ylabel(_tex(f"median (model - truth)/truth  [{_pct()}]"))
    ax[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle(_tex(f"exp54 {label} — curves of growth by HALO-mass tercile "
                      f"(truth solid, model dashed)"), fontsize=12)
    fig.text(0.5, 0.005, _tex(
        "Binned by halo mass, which is a model INPUT. Binning by the truth "
        "stellar mass instead would tilt the residual by (slope - 1) dex per "
        "dex for any conditional-mean model, strongest at small R where this "
        "model's known defect lives."),
        ha="center", va="bottom", fontsize=7.0, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    print("wrote", save_fig(fig, FIGDIR / f"qa_cogmass_{name}")[0], flush=True)


def resid_figure(label, model, truth, mask, name):
    """Residual alone, one panel per epoch, full against fair, with scatter."""
    import matplotlib.pyplot as plt
    set_style()
    fig, ax = plt.subplots(1, 5, figsize=(16.0, 3.4), sharey=True)
    for k in range(len(Z)):
        for sel, c, ls, ttl in ((np.ones(len(mask), bool), "0.45", "-", "all"),
                                (mask[:, k], "#0072B2", "-", "fair")):
            g = sel & np.isfinite(model[:, k, :]).all(axis=1)
            if g.sum() < 10:
                continue
            r = 100 * (model[g, k] - truth[g, k]) / truth[g, k]
            med = np.nanmedian(r, axis=0)
            lo, hi = np.nanpercentile(r, [16, 84], axis=0)
            ax[k].plot(R, med, ls, c=c, lw=1.8,
                       label=_tex(f"{ttl} (n={int(g.sum())})"))
            ax[k].fill_between(R, lo, hi, color=c, alpha=0.13, lw=0)
        ax[k].axhline(0, c="0.6", lw=0.8)
        ax[k].set_xscale("log")
        ax[k].set_xlabel(_tex("R [kpc]"))
        ax[k].set_title(_tex(f"z = {Z[k]}"), fontsize=10)
        ax[k].legend(fontsize=7, loc="lower right")
    ax[0].set_ylabel(_tex(f"(model - truth)/truth  [{_pct()}]"))
    ax[0].set_ylim(-45, 45)
    fig.suptitle(_tex(f"exp54 {label} — residual, all galaxies against the "
                      f"fair sample (band = 16th-84th percentile)"),
                 fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / f"qa_resid_{name}")[0], flush=True)


def main(labels=None):
    import selection as S
    import stage2_multiepoch as s2
    s2._w_init(None)
    pop = np.load(POP)
    all_rows = np.array([g["row"] for g in s2._W["gals"]])
    recs = H.build_records(rows=all_rows, verbose=False)
    sel = np.array([h.row for h in recs])
    truth = pop["data"][sel]
    lmh = pop["logmh"][sel]

    cuts = np.load(SEL, allow_pickle=True)["cuts"]
    m200 = S.sample_masses(np.load(S.HS_NPZ, allow_pickle=True))
    mask = (np.isfinite(m200) & (m200 >= cuts[None, :]))[sel]

    refit = np.load(REFIT, allow_pickle=True)
    perep = np.load(PEREP, allow_pickle=True) if PEREP.exists() else None
    labels = labels or [str(v) for v in refit["labels"]]

    print(f"exp54 QA FIGURES — {len(recs)} galaxies, {len(labels)} models")
    print(f"  fair-sample galaxies per epoch: "
          + " ".join(str(int(v)) for v in mask.sum(0)) + "\n")

    for label in labels:
        for tag, key, src in (("full", f"{label}_theta_full", refit),
                              ("perepoch", f"{label}_theta", perep)):
            if src is None or key not in src:
                continue
            theta = src[key]
            name = f"{label}_{tag}"
            print(f"  {label}  [{tag}-fitted theta]", flush=True)
            model = _predict(label, theta, recs, truth)
            cog_figure(label, model, truth, mask, name)
            cogmass_figure(label, model, truth, lmh, name)
            resid_figure(label, model, truth, mask, name)


if __name__ == "__main__":
    labs = None
    if "--label" in sys.argv:
        labs = [sys.argv[sys.argv.index("--label") + 1]]
    main(labs)
