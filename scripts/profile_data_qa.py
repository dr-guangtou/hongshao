"""QA figures for the consolidated 1-D profile dataset (`hongshao.profile_data`).

Four panels, each answering one question a reader of
``doc/tng300_profile_data.md`` will ask:

  A  Which aperture geometry reproduces the stored curve of growth?
  B  Does the recovered constant isophotal shape match the recorded one?
  C  How large is each term of the error model, radius by radius?
  D  How much of the sample is usable, epoch by epoch?

Run (after ``python -m hongshao.profile_data --build``):

    uv run python scripts/profile_data_qa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hongshao import plotting                                    # noqa: E402
from hongshao.profile_data import (APER6_SMA_KPC, COG_RAD_KPC,   # noqa: E402
                                   EPOCHS, N_EPOCH, REDSHIFTS,
                                   load_profiles)
from hongshao.qa import _zcolors                                 # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
ZLAB = [f"$z={z:g}$" for z in REDSHIFTS]


def _median_abs_log(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(np.log10(np.asarray(num) / np.asarray(den)))
    return np.nanmedian(r, axis=0)


def make_figure(d, out_stem):
    plotting.set_style()
    colors = _zcolors(N_EPOCH)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4))
    axA, axB, axC, axD = axes.ravel()

    # --- A. which geometry reproduces the stored CoG ------------------------
    styles = {"constant shape": dict(ls="-", lw=1.8),
              "variable shape": dict(ls="--", lw=1.2),
              "circular": dict(ls=":", lw=1.2)}
    key = {"constant shape": "cog_const_shape", "variable shape": "cog_var_shape"}
    for k in range(N_EPOCH):
        for name in ("constant shape", "variable shape"):
            axA.plot(COG_RAD_KPC, _median_abs_log(d[key[name]][:, k],
                                                  d["cog_provided"][:, k]),
                     color=colors[k], **styles[name])
        circ = d["cog_const_shape"][:, k] / (1 - d["ellip_const"][:, k, None])
        axA.plot(COG_RAD_KPC, _median_abs_log(circ, d["cog_provided"][:, k]),
                 color=colors[k], **styles["circular"])
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel("aperture semi-major axis [kpc]")
    axA.set_ylabel("median $|\\log_{10}$(rebuilt / stored)$|$ [dex]")
    axA.set_title("A. the stored CoG uses a CONSTANT isophotal shape", loc="left")
    hs = [plt.Line2D([], [], color="0.35", **styles[n]) for n in styles]
    axA.legend(hs, list(styles), loc="lower left", title="rebuilt with",
               ncol=3, fontsize=7)
    axA.set_ylim(top=4e-1)

    # --- B. recovered vs recorded shape at z=0.4 ----------------------------
    et = d["ellip_const_table"][:, 0]
    er = d["ellip_const"][:, 0]
    ok = np.isfinite(et) & np.isfinite(er)
    good = ok & d["iso_shape_ok"][:, 0]
    axB.scatter(et[ok & ~good], er[ok & ~good], s=7, c="#D55E00", alpha=.8,
                label=f"fitter fell back ($n={int((ok & ~good).sum())}$)", zorder=3)
    axB.scatter(et[good], er[good], s=4, c="#0072B2", alpha=.35,
                label=f"clean ($n={int(good.sum())}$)", zorder=2, lw=0)
    lim = [-0.05, 0.75]
    axB.plot(lim, lim, color="0.4", lw=1, ls="--", zorder=1)
    axB.set_xlim(lim)
    axB.set_ylim(lim)
    axB.set_xlabel("ellipticity RECORDED in the aperture table (xy)")
    axB.set_ylabel("ellipticity RECOVERED from stored CoG / circular integral")
    r = np.corrcoef(et[good], er[good])[0, 1]
    bias = np.median(et[good] - er[good])
    rms = np.std(et[good] - er[good])
    axB.set_title(f"B. the shape can be recovered at every epoch\n"
                  f"clean: $r={r:.4f}$, bias ${bias:+.4f}$, rms ${rms:.4f}$",
                  loc="left")
    axB.legend(loc="lower right")

    # --- C. the three-term error model -------------------------------------
    for k in range(N_EPOCH):
        axC.plot(COG_RAD_KPC, np.nanmedian(d["sigma_iso_dex"][:, k], axis=0),
                 color=colors[k], lw=1.6, label=ZLAB[k])
        axC.axhline(np.nanmedian(d["sigma_shape_dex"][:, k]), color=colors[k],
                    lw=0.9, ls=":")
    axC.plot(APER6_SMA_KPC, np.nanmedian(d["sigma_proj_dex"], axis=0),
             color="k", lw=2.2, marker="o", ms=4, ls="--",
             label="projection ($z=0.4$)", zorder=5)
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlabel("aperture semi-major axis [kpc]")
    axC.set_ylabel("$\\sigma$ [dex]")
    axC.set_title("C. three measured error terms; projection dominates\n"
                  "from 10 to 100 kpc at $z=0.4$", loc="left")
    axC.legend(loc="lower left", ncol=2, fontsize=7)
    axC.text(0.97, 0.95, "solid: azimuthal (statistical)\ndotted: constant-shape"
             "\ndashed black: projection", transform=axC.transAxes,
             ha="right", va="top", fontsize=7, color="0.3")

    # --- D. sample quality by epoch ----------------------------------------
    n = int(d["n_galaxies"])
    bars = {
        "isophote list present": (d["n_iso"] > 0).sum(axis=0),
        "flag = True": d["flag_valid"].sum(axis=0),
        "shape fit did not fall back": d["iso_shape_ok"].sum(axis=0),
        "CoG reproduced within 5 per cent": (np.isfinite(d["ellip_const"]) &
                                     (d["cog_over_circ_spread"] < 0.05)).sum(axis=0),
    }
    x = np.arange(N_EPOCH)
    w = 0.2
    for j, (lab, v) in enumerate(bars.items()):
        axD.bar(x + (j - 1.5) * w, np.asarray(v) / n, width=w, label=lab,
                color=plotting.OKABE_ITO[j % len(plotting.OKABE_ITO)])
    axD.set_xticks(x)
    axD.set_xticklabels(ZLAB)
    axD.set_ylim(0, 1.05)
    axD.axhline(1.0, color="0.6", lw=0.8, ls=":")
    axD.set_ylabel(f"fraction of the {n} galaxies")
    axD.set_title("D. usable sample, epoch by epoch", loc="left")
    axD.legend(loc="lower left", fontsize=7)

    fig.tight_layout()
    paths = plotting.save_fig(fig, out_stem)
    plt.close(fig)
    return paths


def main():
    d = load_profiles()
    paths = make_figure(d, FIGDIR / "profile_data_qa")
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
