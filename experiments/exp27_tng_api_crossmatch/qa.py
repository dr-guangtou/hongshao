"""exp27 step 3 — QA figure for the DiffMAH/DiffStar cross-match.

Three panels:
  (a) match-distance histogram: the exact-match peak (~1e-5 cMpc/h) vs the
      off-main-branch tail (~0.02-0.12 cMpc/h) that gets flagged matched=False;
  (b) consistency of the input z=0.4 halo mass against the official main-branch
      M200c at snap 72 (matched galaxies only) -- a tight diagonal validates the
      bridge end-to-end;
  (c) a few example MAHs: official SUBFIND main-branch M200c vs the DiffMAH
      `log_mah_sim` (running-max Mpeak it was fit to) vs the DiffMAH fit curve --
      shows the merger-tail bias that motivates the summed-accreted MAH.

Run: PYTHONPATH=. uv run python experiments/exp27_tng_api_crossmatch/qa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
from astropy.table import Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hongshao.plotting import set_style, save_fig, OKABE_ITO  # noqa: E402

set_style()
OUT, FIG = HERE / "outputs", HERE / "figures"
SNAP_Z04 = 72


def main() -> None:
    t = Table.read(OUT / "crossmatch.fits")
    mah = np.load(OUT / "official_mah.npz")
    loaded = np.isfinite(t["match_dist_cmpc"])
    matched = np.asarray(t["matched"])
    tgyr = mah["cosmic_time_gyr"]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))

    # (a) match-distance histogram
    d = np.asarray(t["match_dist_cmpc"])[loaded]
    d = np.clip(d, 1e-6, None)
    ax[0].hist(np.log10(d), bins=50, color=OKABE_ITO[4])
    ax[0].axvline(np.log10(1e-3), ls="--", c="k", lw=1)
    ax[0].set(xlabel=r"$\log_{10}$ match distance [cMpc/$h$]",
              ylabel="galaxies",
              title=f"(a) match quality  ({matched.sum()}/{loaded.sum()} matched)")
    ax[0].text(np.log10(1e-3) + 0.1, ax[0].get_ylim()[1] * 0.7,
               "tol = 1 ckpc/$h$", fontsize=7)

    # (b) input z=0.4 halo mass vs official main-branch M200c at snap 72
    m200_72 = mah["log_m200c"][:, SNAP_Z04]
    sel = matched & np.isfinite(m200_72)
    x, y = np.asarray(t["logmh_z0p4"])[sel], m200_72[sel]
    ax[1].scatter(x, y, s=4, alpha=0.3, color=OKABE_ITO[5], edgecolors="none")
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set(xlabel=r"input $\log M_{h}$ (z=0.4)",
              ylabel=r"official $\log M_{\rm 200c}$ (snap 72)",
              title=f"(b) mass consistency  (median $\\Delta$={np.median(y-x):+.3f} dex)")

    # (c) example MAHs: official M200c vs DiffMAH sim (Mpeak) vs DiffMAH fit
    ex = np.where(matched & np.isfinite(mah["log_m200c"]).any(axis=1))[0][:4]
    snaps = np.arange(100)
    xaxis = tgyr if tgyr.size == 100 else snaps
    xlab = "cosmic time [Gyr]" if tgyr.size == 100 else "snapshot"
    for k, i in enumerate(ex):
        c = OKABE_ITO[k]
        ax[2].plot(xaxis, mah["log_m200c"][i], "-", c=c, lw=1.5,
                   label=f"gal {t['index'][i]}" if k == 0 else None)
        ax[2].plot(xaxis, mah["diffmah_log_mah_sim"][i], ":", c=c, lw=1.2)
        ax[2].plot(xaxis, mah["diffmah_log_mah_fit"][i], "--", c=c, lw=1)
    ax[2].plot([], [], "k-", label="official M200c")
    ax[2].plot([], [], "k:", label="DiffMAH sim (Mpeak)")
    ax[2].plot([], [], "k--", label="DiffMAH fit")
    ax[2].set(xlabel=xlab, ylabel=r"$\log M\ [M_\odot]$",
              title="(c) example MAHs (4 galaxies)", ylim=(11, 15.4))
    ax[2].legend(fontsize=7)

    fig.tight_layout()
    paths = save_fig(fig, FIG / "crossmatch_qa")
    print("wrote", paths[0])


if __name__ == "__main__":
    sys.exit(main())
