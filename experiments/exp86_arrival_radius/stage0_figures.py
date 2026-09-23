"""exp86 — the one figure (`figures/qa/exp86_probe_summary`):
  (a) the 50-100 kpc shell's median bias at z = 1.5 and z = 2 on the fitting
      sample and on the halo-mass-complete progenitors, for the adopted
      mean, the re-tuned control and each strength of arrival sizing: the
      two samples' opposite-signed biases never move;
  (b) the arrival factor log R200c(t_a)/R200c(t') of the shell's deposits at
      z = 2, complete progenitors vs the rest: the knob has 0.013 dex to
      separate them with, against a 0.20 dex gap in the shell residual;
  (c) per galaxy at z = 2: the shell residual against the arrival factor,
      by sample — a real but shallow trend inside each sample, and the
      offset between samples that is the selection.
Reads `outputs/stage0_probe.npz`; no build.
Run: uv run python -u experiments/exp86_arrival_radius/stage0_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hongshao import qa                                  # noqa: E402

FIGDIR = HERE / "figures/qa"                             # the experiment's own figures, never the repo-level figures/qa
C_FIT, C_MHC = "#0072B2", "#D55E00"


def main():
    z = np.load(HERE / "outputs/stage0_probe.npz", allow_pickle=True)
    keys = [str(k) for k in z["gate_keys"]]
    probes = [str(p) for p in z["probe_keys"]]
    G = np.asarray(z["gates"], float)
    tab = {p: dict(zip(keys, G[i])) for i, p in enumerate(probes)}
    plt.rcParams.update({"axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    ax = axes[0]
    order = ["adopted", "control"] + sorted((p for p in probes if p.startswith("w_arr")), key=lambda p: float(p.split("=")[1]))
    labels = ["adopted\nmean", "control\n(re-tuned)"] + [p.replace("w_arr = ", "w = ") for p in order[2:]]
    x = np.arange(len(order))
    for k, zz, ls in ((3, "1.5", "-"), (4, "2", "--")):
        ax.plot(x, [tab[p][f"shell_fit_{k}"] for p in order], marker="o", color=C_FIT, ls=ls, lw=1.6, label=f"fitting sample, z = {zz}")
        ax.plot(x, [tab[p][f"shell_mhc_{k}"] for p in order], marker="s", color=C_MHC, ls=ls, lw=1.6, label=f"halo-mass-complete, z = {zz}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("50" + qa._tex("-") + "100 kpc shell, median (model " + qa._tex("-") + " data)/data [" + qa._pct() + "]")
    ax.set_title("(a) the probe: neither sample's shell moves", fontsize=11)
    ax.legend(frameon=False, loc="center left")
    ax.set_ylim(-25, 12)

    fac = np.asarray(z["arrival_factor"], float)[:, 4]
    comp = np.asarray(z["complete"], bool)[:, 4]
    sh = (np.asarray(z["shell_model_c"], float) + np.asarray(z["shell_model_e"], float))[:, 4]
    st = np.asarray(z["shell_truth"], float)[:, 4]
    ok = np.isfinite(fac) & np.isfinite(st) & (st > 0)
    res = np.full_like(fac, np.nan)
    res[ok] = np.log10(sh[ok] / st[ok])

    ax = axes[1]
    bins = np.linspace(0.05, 0.25, 41)
    ax.hist(fac[ok & ~comp], bins=bins, color=C_FIT, alpha=0.6, density=True, label="the rest of the fitting sample")
    ax.hist(fac[ok & comp], bins=bins, color=C_MHC, alpha=0.6, density=True, label="halo-mass-complete progenitors")
    for sel, c in ((ok & ~comp, C_FIT), (ok & comp, C_MHC)):
        ax.axvline(np.median(fac[sel]), color=c, lw=1.6)
    ax.set_xlabel(qa._tex("arrival factor of the shell's deposits, log R200c(t_a) / R200c(t') [dex], z = 2"))
    ax.set_ylabel("density")
    ax.set_title(f"(b) the knob's lever: medians differ by {np.median(fac[ok & comp]) - np.median(fac[ok & ~comp]):.3f} dex", fontsize=11)
    ax.legend(frameon=False)

    ax = axes[2]
    for sel, c, lab in ((ok & ~comp, C_FIT, "the rest"), (ok & comp, C_MHC, "halo-mass-complete")):
        ax.scatter(fac[sel], res[sel], s=5, color=c, alpha=0.3, rasterized=True, label=lab)
        q = np.quantile(fac[sel], np.linspace(0, 1, 9))
        xm = [np.median(fac[sel][(fac[sel] >= q[i]) & (fac[sel] <= q[i + 1])]) for i in range(8)]
        ym = [np.median(res[sel][(fac[sel] >= q[i]) & (fac[sel] <= q[i + 1])]) for i in range(8)]
        ax.plot(xm, ym, color=c, lw=2.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(qa._tex("arrival factor [dex], z = 2"))
    ax.set_ylabel(qa._tex("shell residual log10(model / data) [dex], z = 2"))
    ax.set_title("(c) the gap between samples is the selection, not the factor", fontsize=11)
    ax.legend(frameon=False, loc="lower left")
    ax.set_xlim(0.05, 0.25)
    ax.set_ylim(-1.0, 1.0)
    fig.suptitle("exp86: sizing the delayed deposits by the halo at arrival cannot fix the complete progenitors' 50" + qa._tex("-") + "100 kpc shell",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"exp86_probe_summary.{ext}", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGDIR / 'exp86_probe_summary.png'}")


if __name__ == "__main__":
    main()
