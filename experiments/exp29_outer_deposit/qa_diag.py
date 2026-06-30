"""exp29 — QA diagnostics: what the RMS metric measures, train vs test, and the
per-galaxy ceiling vs the universal forward model.

Metric (per galaxy): RMS over the 5 epochs x 24 CoG radii of
    log10 model_CoG(R, z_k) - log10 measured_CoG(R, z_k),
i.e. the dex error of the cumulative profile, averaged over radius and epoch.
Reported as the median over galaxies.

Figures:
  - exp29_qa_rms_hist: distribution of that RMS for the shared WF model on TRAIN vs
    TEST, and for the per-galaxy best fit (the ceiling) -> shows train~test (not
    overfitting) and how much the per-galaxy freedom buys.
  - exp29_qa_train_vs_pergal: example TRAIN galaxies, shared-forward (solid) vs
    per-galaxy fit (dashed) vs TNG (dots) -> separates "model can't fit" from
    "universality costs".

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/qa_diag.py
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
from phase4d import standardize, theta_of                                          # noqa: E402
from run import ANCHOR_Z                                                            # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                          # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                        # noqa: E402

set_style()
FIGDIR = HERE / "figures"
R = COG_RAD_KPC
CACHE = HERE / "outputs" / "p4d_cache.npz"


def main():
    gals = load_pop(200); standardize(gals)
    rng = np.arange(len(gals)); train = rng[rng % 2 == 0]; test = rng[rng % 2 == 1]
    z = np.load(CACHE, allow_pickle=True)
    pars, WFp, WFm = z["pars"], z["p_WF"], str(z["m_WF"])

    rms_wf = np.array([gal_rms(theta_of(WFp, gals[i], WFm), gals[i]) for i in rng])
    rms_pg = np.array([gal_rms(pars[i], gals[i]) for i in rng])
    print(f"median CoG RMS [dex]:  WF shared train {np.median(rms_wf[train]):.3f}  "
          f"test {np.median(rms_wf[test]):.3f}   |  per-galaxy {np.median(rms_pg):.3f}")

    # Fig 1 — RMS distributions
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 0.3, 31)
    ax.hist(rms_wf[train], bins, alpha=0.5, color=OKABE_ITO[0], label=f"WF shared, train (med {np.median(rms_wf[train]):.3f})")
    ax.hist(rms_wf[test], bins, alpha=0.5, color=OKABE_ITO[1], label=f"WF shared, test (med {np.median(rms_wf[test]):.3f})")
    ax.hist(rms_pg, bins, histtype="step", lw=2, color="k", label=f"per-galaxy fit (med {np.median(rms_pg):.3f})")
    ax.set(xlabel="per-galaxy 5-epoch CoG RMS [dex]", ylabel="count",
           title="exp29 QA — error distribution: shared forward (train≈test) vs per-galaxy ceiling")
    ax.legend(fontsize=9); fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "exp29_qa_rms_hist")[0])

    # Fig 2 — train galaxies: shared-forward vs per-galaxy fit vs TNG
    sel = list(train[np.argsort([gals[i]["logMstar"] for i in train])[::-1]][::max(1, len(train) // 6)][:6])
    cmap = matplotlib.colormaps["viridis"]
    fig2, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    for ax, i in zip(axes.ravel(), sel):
        g = gals[i]; pf = predict(WFp if False else theta_of(WFp, g, WFm), g); pg = predict(pars[i], g)
        for k, zz in enumerate(ANCHOR_Z):
            c = cmap(k / 4)
            ax.plot(R, g["logC"][k], "o", c=c, ms=3, alpha=0.7)
            ax.plot(R, np.log10(np.clip(pf[k], 1, None)), "-", c=c, lw=1.5)
            ax.plot(R, np.log10(np.clip(pg[k], 1, None)), "--", c=c, lw=1.0)
        ax.set(xscale="log", xlabel="R [kpc]", ylabel=r"$\log M_*(<R)$",
               title=f"logM*={g['logMstar']:.2f} logMh={g['logMh']:.2f}  "
                     f"WF={gal_rms(theta_of(WFp, g, WFm), g):.3f} / pg={gal_rms(pars[i], g):.3f}")
    axes[0, 0].plot([], [], "-", c="k", label="shared forward (WF)")
    axes[0, 0].plot([], [], "--", c="k", label="per-galaxy fit")
    axes[0, 0].plot([], [], "o", c="k", label="TNG")
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig2.suptitle("exp29 QA — TRAIN galaxies: shared forward (solid) vs per-galaxy fit (dashed) "
                  "vs TNG (dots); z=0.4..2.0 dark->light", fontsize=12)
    fig2.tight_layout()
    print("wrote", save_fig(fig2, FIGDIR / "exp29_qa_train_vs_pergal")[0])


if __name__ == "__main__":
    sys.exit(main())
