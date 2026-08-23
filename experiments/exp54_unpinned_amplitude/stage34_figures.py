"""exp54 Stage 3.4 — figures for the profile-shape representational ceiling.

Two figures, and each exists to stop a specific misreading.

  `stage34_ladder_<label>`  the FREEDOM LADDER. Left panel: the production
      shape loss of each bound, epoch by epoch. Remaining panels: where in
      radius each bound's residual sits. The misreading it prevents is
      "the ceiling is far below the model, so a better efficiency law will
      close the gap" -- the same panel carries the PERMUTED control, which is
      this galaxy's deposits fitted to ANOTHER galaxy's measured profile. Where
      the permuted line sits close to the ceiling, the ceiling's advantage is
      generic flexibility that no global law can supply.

  `stage34_cz_<label>`      the SCAN of a redshift-dependent deposit shape,
      `c = c0 + c_z ln(1+z)`, at two levels of per-galaxy freedom. The
      misreading it prevents is "the parameter was never tried": it was tried
      at the level of the BOUND, where it cannot be rescued by refitting
      something else, and the optimum is at `c_z = 0` in both panels.

TERMS ON THE AXES, none of which should be read as obvious:

  **score_F**  the production shape loss divided by `L_F_ref = 0.158196`, the
    previous-generation model's value, so 1.0 means "as good as the incumbent"
    and lower is better. The loss itself is the root-mean-square fractional
    residual of `M*(<R)/M*(<100 kpc)` over the 24 radii, averaged over galaxies.

  **rung**  one bound in the ladder, distinguished by how much freedom it is
    given PER GALAXY: `monotone` none but the requirement that enclosed mass
    never falls with time; `epoch-alone` a separate weight vector at every
    epoch; `ceiling` one causal weight vector of 71 non-negative numbers;
    `interval5` five numbers, one per interval between observation epochs;
    `model` the fitted seven global parameters and nothing per galaxy.

  **mh-complete sample**  each epoch's galaxies whose halo is massive enough
    that being the progenitor of a z=0.4 selection is no longer a strong extra
    condition. 2397 at z=0.4 falling to 840 at z=2.

Called by `stage34_ceiling.main(figures=True)`; also runnable directly against
the saved outputs:
    HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python -u \\
    experiments/exp54_unpinned_amplitude/stage34_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", HERE):
    sys.path.insert(0, str(p))

# imported before `halo`/`model` bind other experiments' same-named siblings
import fit as _fit_import_order_guard                    # noqa: E402,F401
import stage34_ceiling as C                              # noqa: E402
from hongshao.plotting import save_fig, set_style        # noqa: E402
from hongshao.qa import _pct, _tex                       # noqa: E402

R = C.R
Z = np.array(C.Z)
FIGDIR = HERE / "figures"

#: one colour per rung, ordered from most to least per-galaxy freedom.
#: Okabe-Ito, chosen so the two bounds that bracket the model (`ceiling` and
#: `interval5`) are the two most distinct hues.
RUNG_STYLE = {
    "monotone":    ("#999999", "-", 1.6, "monotone-in-time floor (basis-free)"),
    "epoch-alone": ("#56B4E9", "-", 1.6, "each epoch fitted alone (not causal)"),
    "ceiling":     ("#0072B2", "-", 2.4, "ceiling: 71 causal weights/galaxy"),
    "ceiling_fb":  ("#0072B2", ":", 1.4, r"the same with $\varepsilon \leq f_b$"),
    "interval5":   ("#E69F00", "-", 2.0, "5 interval weights/galaxy"),
    "model":       ("#D55E00", "-", 2.4, "fitted model: 7 global parameters"),
    "permuted":    ("#009E73", "--", 1.6, "ceiling fitted to ANOTHER galaxy"),
}


def _caption(fig, text, width=190, fontsize=7.0):
    """Lay a wrapped caption under the axes and return the rect to leave for it.

    matplotlib's own `wrap=True` measures against the figure width, but the
    house style saves with `bbox_inches="tight"`, so an over-long line silently
    widens the canvas instead of wrapping. Wrapping the string first keeps the
    saved aspect ratio equal to the requested `figsize`.
    """
    import textwrap
    lines = textwrap.wrap(text, width=width)
    fig.text(0.5, 0.004, _tex("\n".join(lines)), ha="center", va="bottom",
             fontsize=fontsize, style="italic", color="0.35",
             linespacing=1.45)
    h = fig.get_size_inches()[1]
    return 0.010 + len(lines) * (fontsize * 1.45 / 72.0) / h


def ladder_figure(label, res, data, mask, name=None):
    """score_F per epoch for every rung, and where each one's residual sits."""
    import matplotlib.pyplot as plt
    set_style()
    name = name or label
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.0))
    ax = ax.ravel()

    for key in ("monotone", "epoch-alone", "ceiling", "ceiling_fb",
                "interval5", "model", "permuted"):
        col, ls, lw, lab = RUNG_STYLE[key]
        sf = res["scores"][key][1]
        ax[0].plot(Z, sf, ls, c=col, lw=lw, marker="o", ms=3.5,
                   label=_tex(lab))
    ax[0].axhline(1.0, c="0.75", lw=0.8, ls=":")
    ax[0].text(1.98, 1.01, _tex("the previous-generation model"), fontsize=6.5,
               color="0.5", va="bottom", ha="right")
    ax[0].set_xlabel(_tex("redshift z"))
    ax[0].set_ylabel(_tex("score_F   (shape loss / L_F_ref; lower is better)"))
    ax[0].set_title(_tex("A. the freedom ladder"), fontsize=10)
    ax[0].set_ylim(0.0, 1.62)
    ax[0].legend(fontsize=6.6, loc="upper center", ncol=2, framealpha=0.0,
                 frameon=False, handlelength=2.4, columnspacing=1.2)

    worst = 0.0
    for k in range(5):
        a = ax[k + 1]
        for key in ("monotone", "epoch-alone", "ceiling", "interval5", "model"):
            col, ls, lw, lab = RUNG_STYLE[key]
            g = mask[:, k]
            med = 100.0 * np.median(res["preds"][key][g, k]
                                    / data[g, k] - 1.0, axis=0)
            worst = max(worst, float(np.nanmax(np.abs(med))))
            a.plot(R, med, ls, c=col, lw=lw)
        a.axhline(0, c="0.6", lw=0.8)
        for y in (-10, 10):
            a.axhline(y, c="0.88", lw=0.7, ls=":")
        a.set_xscale("log")
        a.set_xlabel(_tex("R [kpc]"))
        a.set_title(_tex(f"z = {Z[k]}   (n = {int(mask[:, k].sum())})"),
                    fontsize=9.5)
        if k in (0, 2):
            a.set_ylabel(_tex(f"median (prediction - truth)/truth  [{_pct()}]"))
    lim = float(np.clip(1.15 * worst, 8.0, 60.0))       # adaptive, not fixed
    for a in ax[1:]:
        a.set_ylim(-lim, lim)

    fig.suptitle(_tex(f"exp54 {label} — the profile-shape representational "
                      f"ceiling: what limits the model, and by how much"),
                 fontsize=12)
    bot = _caption(fig, (
        "Every curve is a BOUND on the deposition-only framework, differing "
        "only in how much freedom it is given per galaxy; all are scored on "
        "each epoch's own mh-complete galaxies. Read the ladder from the "
        "bottom: the basis draws any single epoch's profile almost exactly "
        "(`epoch-alone`), so the profile family and the size law are not the "
        "limitation. Requiring ONE causal weight vector to serve all five "
        "epochs costs a factor of five (`ceiling`), and the fitted model sits "
        "close to what a per-galaxy oracle over the coarse time distribution "
        "of deposited mass can reach (`interval5`). The green dashed line is "
        "the same 71-weight solve fitted to a DIFFERENT galaxy's measured "
        "profile: where it lies near the ceiling, the ceiling's margin is "
        "generic flexibility and not information a global law could use."))
    fig.tight_layout(rect=(0, bot, 1, 1))
    print("wrote", save_fig(fig, FIGDIR / f"stage34_ladder_{name}")[0],
          flush=True)


def cz_figure(scan_npz, label="gompertz_log-E2-S2", name=None):
    """The redshift-dependent deposit shape, scanned at two freedom levels."""
    import matplotlib.pyplot as plt
    set_style()
    name = name or label
    z = np.load(scan_npz, allow_pickle=True)
    fitted = z["fitted"]
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.6))

    for i, (key, ttl) in enumerate((
            ("scan_shape", "B. at the 71-weight ceiling"),
            ("scan_shape_i5", "C. at the 5-weight `interval5` rung"))):
        rec = z[key]
        c0s, czs = np.unique(rec[:, 2]), np.unique(rec[:, 3])
        grid = np.full((len(c0s), len(czs)), np.nan)
        for row in rec:
            grid[np.searchsorted(c0s, row[2]), np.searchsorted(czs, row[3])] = row[4]
        # the surface spans a factor of four, so a linear scale hides the
        # structure that matters -- everything near the minimum. Saturate at
        # the median and mark the colour bar as extended, rather than
        # pretending the yellow plateau is uniform.
        vmin = np.nanmin(grid)
        im = ax[i].pcolormesh(czs, c0s, grid, cmap="cividis_r", vmin=vmin,
                              vmax=float(np.nanmedian(grid)), shading="nearest")
        fig.colorbar(im, ax=ax[i], extend="max",
                     label=_tex("ceiling in score_F (saturated at the median)"))
        b = rec[np.argmin(rec[:, 4])]
        ax[i].plot(b[3], b[2], "*", c="#F0E442", ms=17, mec="k", mew=0.9,
                   label=_tex(f"grid optimum: c_z = {b[3]:+.2f}"))
        ax[i].plot(0.0, fitted[2], "o", mfc="none", mec="k", mew=1.4, ms=13,
                   label=_tex(f"fitted model: c = {fitted[2]:.3f}, c_z = 0"))
        ax[i].set_xlabel(r"$c_z$" + _tex("  in  c = c0 + c_z ln(1+z)"))
        ax[i].set_ylabel(r"$c_0$")
        ax[i].set_title(_tex(ttl), fontsize=10)
        ax[i].legend(fontsize=7, loc="upper left", framealpha=0.9,
                     frameon=True, facecolor="white", edgecolor="0.8")

    rec = z["scan_shape"]
    rec5 = z["scan_shape_i5"]
    # the grid stores c0 rounded to 4 dp, so match on the NEAREST grid value
    c0_cut = np.unique(rec[:, 2])[np.argmin(np.abs(np.unique(rec[:, 2])
                                                   - fitted[2]))]
    sel = np.abs(rec[:, 2] - c0_cut) < 1e-9
    sel5 = np.abs(rec5[:, 2] - c0_cut) < 1e-9
    ax[2].plot(rec[sel, 3], rec[sel, 4], "-o", c="#0072B2", ms=3.5,
               label=_tex("ceiling (71 weights)"))
    ax[2].plot(rec[sel, 3], rec[sel, 5], "--", c="#0072B2", lw=1.2,
               label=_tex("its permuted control"))
    ax[2].plot(rec5[sel5, 3], rec5[sel5, 4], "-o", c="#E69F00", ms=3.5,
               label=_tex("interval5 (5 weights)"))
    ax[2].plot(rec5[sel5, 3], rec5[sel5, 5], "--", c="#E69F00", lw=1.2,
               label=_tex("its permuted control"))
    ax[2].axvline(0.0, c="0.6", lw=0.9, ls=":")
    ax[2].set_xlabel(r"$c_z$")
    ax[2].set_ylabel(_tex("score_F  (lower is better)"))
    ax[2].set_title(_tex(f"D. cut at the fitted c0 = {c0_cut:.3f}"),
                    fontsize=10)
    ax[2].legend(fontsize=7)

    fig.suptitle(_tex(f"exp54 {label} — would a redshift-dependent deposit "
                      f"shape raise the ceiling? No, at either freedom level"),
                 fontsize=12)
    bot = _caption(fig, (
        "Each grid point gets exactly the same per-galaxy freedom; only the "
        "GLOBAL deposit-shape parameters move, so a drop in the surface would "
        "be a statement about the basis rather than about degrees of freedom. "
        "The parameter was proposed against the model's high-redshift "
        "diffuseness, on the reasoning that the basis cannot make compact "
        "enough deposits at early times. The optimum is at c_z = 0 in both "
        "panels and the surface rises in both directions, so the extension "
        "cannot lower the bound -- and a term that cannot lower the bound "
        "cannot lower the fitted loss by the mechanism it was proposed for."))
    fig.tight_layout(rect=(0, bot, 1, 1))
    print("wrote", save_fig(fig, FIGDIR / f"stage34_cz_{name}")[0], flush=True)


if __name__ == "__main__":
    import stage2_multiepoch as s2
    import halo as H
    s2._w_init(None)
    pop = np.load(C.POP)
    saved = np.load(C.OUT, allow_pickle=True)
    rows, mask = saved["rows"], saved["mask"]
    data = pop["data"][rows]
    recs = H.build_records(rows=rows, verbose=False)
    for label in [str(v) for v in saved["labels"]]:
        res = dict(scores={k: (saved[f"{label}_{k}_sA"], saved[f"{label}_{k}_sF"])
                           for k in list(C.LADDER) + ["permuted",
                                                      "truth-vs-truth"]},
                   preds={k: saved[f"{label}_pred_{k}"] for k in C.LADDER})
        ladder_figure(label, res, data, mask)
    scan = HERE / "outputs" / "stage34_basis_scan.npz"
    if scan.exists():
        cz_figure(scan)
