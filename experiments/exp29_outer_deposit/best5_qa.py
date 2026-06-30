"""exp29 — does the log-RMS metric reflect the actual per-galaxy fit quality?

Pick the 5 galaxies with the LOWEST per-galaxy-fit RMS ("best performance") and show
their 5-epoch curve of growth in LINEAR stellar mass M*(<R) (data vs best-fit model)
plus the relative residual (model - data)/data. If a low log-RMS still hides large
or structured relative errors, the metric is not telling the whole story.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/best5_qa.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from population import load_pop, predict, gal_rms                                   # noqa: E402
from run import ANCHOR_Z                                                            # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                          # noqa: E402
from hongshao.plotting import set_style, save_fig                                   # noqa: E402

set_style()
FIGDIR = HERE / "figures"
R = COG_RAD_KPC
CACHE = HERE / "outputs" / "p4d_cache.npz"


def main():
    gals = load_pop(200)
    pars = np.load(CACHE, allow_pickle=True)["pars"]                 # per-galaxy best-fit params
    rms = np.array([gal_rms(pars[i], gals[i]) for i in range(len(gals))])
    best = np.argsort(rms)[:5]
    print("5 lowest per-galaxy-fit RMS galaxies:")
    for i in best:
        g = gals[i]; pred = predict(pars[i], g); data = 10.0 ** g["logC"]
        rel = (pred - data) / data
        print(f"  logM*={g['logMstar']:.2f} logMh={g['logMh']:.2f}  RMS(log)={rms[i]:.3f}  "
              f"max|rel|={np.abs(rel).max():.3f}  median|rel|={np.median(np.abs(rel)):.3f}")

    cmap = matplotlib.colormaps["viridis"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.4), sharex=True)
    for col, i in enumerate(best):
        g = gals[i]; pred = predict(pars[i], g); data = 10.0 ** g["logC"]
        aC, aR = axes[0, col], axes[1, col]
        for k, z in enumerate(ANCHOR_Z):
            c = cmap(k / 4)
            aC.plot(R, data[k] / 1e11, "o", c=c, ms=3.5, alpha=0.8)
            aC.plot(R, pred[k] / 1e11, "-", c=c, lw=1.6, label=f"z={z}")
            aR.plot(R, 100 * (pred[k] - data[k]) / data[k], "-", c=c, lw=1.5)
        aC.set(xscale="log", title=f"logM*={g['logMstar']:.2f}  RMS={rms[i]:.3f}")
        aR.set(xscale="log", xlabel="R [kpc]", ylim=(-25, 25))
        aR.axhline(0, c="0.6", lw=0.8)
        if col == 0:
            aC.set_ylabel(r"$M_*(<R)\ [10^{11}\,M_\odot]$"); aR.set_ylabel(r"(model$-$data)/data [%]")
            aC.legend(fontsize=7, loc="upper left")
    fig.suptitle("exp29 — 5 best per-galaxy fits: curve of growth in LINEAR M* (top; dots=TNG, "
                 "lines=model) and relative residual (bottom); z=0.4 (dark) to z=2.0 (light)", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp29_best5_linear")[0])

    # complementary: where do the per-galaxy fits systematically fail? (stacked rel residual)
    rel = np.array([(predict(pars[i], gals[i]) - 10.0 ** gals[i]["logC"]) / 10.0 ** gals[i]["logC"]
                    for i in range(len(gals))])              # (Ngal, 5, 24)
    fig2, ax = plt.subplots(figsize=(9, 5.4))
    for k, z in enumerate(ANCHOR_Z):
        c = cmap(k / 4); med = 100 * np.median(rel[:, k], 0)
        ax.plot(R, med, "-", c=c, lw=2, label=f"z={z}")
        if k in (0, 4):
            lo, hi = 100 * np.percentile(rel[:, k], [16, 84], axis=0)
            ax.fill_between(R, lo, hi, color=c, alpha=0.15)
    ax.axhline(0, c="0.6", lw=0.8); ax.axvspan(R.min(), 3, color="0.85", alpha=0.5)
    ax.text(2.05, 18, "inner 3 kpc\n(softening)", fontsize=7, c="0.4")
    ax.set(xscale="log", xlabel="R [kpc]", ylabel="median (model-data)/data [%]", ylim=(-22, 22),
           title="exp29 — stacked relative residual of the per-galaxy fits (n=200): structured, "
                 "worst at inner R and high z")
    ax.legend(fontsize=8, title="epoch (band=16-84 for z0.4,z2)")
    fig2.tight_layout(); print("wrote", save_fig(fig2, FIGDIR / "exp29_residual_map")[0])


if __name__ == "__main__":
    sys.exit(main())
