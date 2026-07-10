"""exp30 phase 4 QA — direct CoG visualization for the population emulator.

The standard QA (profile max|rel| + mass_qa aperture tables) summarizes; this shows
the curves. Two figures per config, from the LOGO universal-theta predictions:
  1. median data vs model CoG in three stellar-mass terciles, with the median
     relative-residual profile per epoch underneath;
  2. best/worst gallery — the 2 best and 2 worst galaxies by LOGO epoch-avg
     max|rel|, CoGs + residuals, worst radius marked (the metric's location).

Run: PYTHONPATH=. uv run python experiments/exp30_transport_kernel/profile_qa.py
Demo: ... profile_qa.py demo
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
from run import ANCHOR_Z                                                             # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                            # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR, OUTDIR = HERE / "figures", HERE / "outputs"
R = COG_RAD_KPC
ZCOL = [matplotlib.colormaps["cividis"](v) for v in np.linspace(0.0, 0.92, 5)]


def load_config(config):
    """Align phase-4 LOGO predictions with data/logms; drop failed (NaN) folds.
    Returns (cogs (n,5,24), datas (n,5,24), logms (n,), index (n,), mr (n,5))."""
    dp = np.load(OUTDIR / "param_emulator.npz")
    d4 = np.load(OUTDIR / "pop_forward.npz")
    row = {int(g): i for i, g in enumerate(dp["index"])}
    sel = np.array([row[int(g)] for g in d4[config + "_index"]])
    cogs, mr = d4[config + "_cogs_univ"], d4[config + "_mr_univ"]
    ok = np.isfinite(cogs).all(axis=(1, 2))
    return (cogs[ok], dp["data"][sel][ok], dp["logms"][sel][ok],
            d4[config + "_index"][ok], mr[ok])


def _cog_panel(ax, rax, cogs_set, datas_set, title):
    """Median log-CoG (data solid, model dashed) + median residual, per epoch."""
    for k in range(5):
        med_d = np.median(np.log10(datas_set[:, k]), axis=0)
        med_m = np.median(np.log10(cogs_set[:, k]), axis=0)
        rel = 100 * np.median((cogs_set[:, k] - datas_set[:, k]) / datas_set[:, k],
                              axis=0)
        ax.plot(R, med_d, "-", c=ZCOL[k], lw=1.8, label=f"z={ANCHOR_Z[k]}")
        ax.plot(R, med_m, "--", c=ZCOL[k], lw=1.4)
        rax.plot(R, rel, "-", c=ZCOL[k], lw=1.4)
    rax.axhline(0, c="0.6", lw=0.8)
    for y in (-20, 20):
        rax.axhline(y, c="0.8", lw=0.7, ls=":")
    ax.set(xscale="log", title=title)
    rax.set(xscale="log", xlabel="R [kpc]", ylim=(-60, 60))


def median_bins_figure(config, cogs, datas, logms):
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.5), sharex=True,
                             height_ratios=[2, 1])
    for b in range(3):
        m = (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
        _cog_panel(axes[0, b], axes[1, b], cogs[m], datas[m],
                   f"logM* {edges[b]:.2f}-{edges[b+1]:.2f}  (n={m.sum()})")
    axes[0, 0].set(ylabel="median log$_{10}$ M$_*$(<R) [M$_\\odot$]")
    axes[1, 0].set(ylabel="median (model$-$data)/data [%]")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"exp30 phase 4 QA [{config}] — LOGO universal-theta CoGs by "
                 "stellar-mass tercile (data solid, model dashed)", fontsize=12)
    fig.tight_layout()
    return save_fig(fig, FIGDIR / f"exp30_profile_qa_bins_{config}")[0]


def cases_figure(config, cogs, datas, logms, index, mr):
    order = np.argsort(mr.mean(axis=1))
    picks = list(order[:2]) + list(order[-2:])
    tags = ["best 1", "best 2", "worst 2", "worst 1"]
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 7.5), sharex=True,
                             height_ratios=[2, 1])
    for c, (i, tag) in enumerate(zip(picks, tags)):
        ax, rax = axes[0, c], axes[1, c]
        for k in range(5):
            rel = 100 * (cogs[i, k] - datas[i, k]) / datas[i, k]
            ax.plot(R, np.log10(datas[i, k]), "-", c=ZCOL[k], lw=1.8,
                    label=f"z={ANCHOR_Z[k]}" if c == 0 else None)
            ax.plot(R, np.log10(cogs[i, k]), "--", c=ZCOL[k], lw=1.4)
            rax.plot(R, rel, "-", c=ZCOL[k], lw=1.4)
            j = np.argmax(np.abs(rel))                    # the metric's radius
            rax.plot(R[j], rel[j], "o", c=ZCOL[k], ms=5, mec="0.2", mew=0.5)
        rax.axhline(0, c="0.6", lw=0.8)
        ax.set(xscale="log", title=f"{tag}: gal {int(index[i])}, "
               f"logM*={logms[i]:.2f}\nepoch-avg max|rel| {100*mr[i].mean():.0f}%")
        rax.set(xscale="log", xlabel="R [kpc]")
    axes[0, 0].set(ylabel="log$_{10}$ M$_*$(<R) [M$_\\odot$]")
    axes[1, 0].set(ylabel="(model$-$data)/data [%]")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"exp30 phase 4 QA [{config}] — best/worst LOGO cases "
                 "(data solid, model dashed; dot = worst radius, the max|rel| metric)",
                 fontsize=12)
    fig.tight_layout()
    return save_fig(fig, FIGDIR / f"exp30_profile_qa_cases_{config}")[0]


def main():
    for config in ("real", "diffmah"):
        cogs, datas, logms, index, mr = load_config(config)
        print(f"[{config}] n={len(cogs)}  epoch-avg max|rel| median "
              f"{100*np.median(mr.mean(1)):.1f}%  "
              f"range {100*mr.mean(1).min():.0f}-{100*mr.mean(1).max():.0f}%")
        print("wrote", median_bins_figure(config, cogs, datas, logms))
        print("wrote", cases_figure(config, cogs, datas, logms, index, mr))


def demo():
    """Self-check: alignment sane; pinned models have ~0 residual at the outermost
    radius; the stored mr matches a recomputation from the curves."""
    cogs, datas, logms, index, mr = load_config("real")
    assert len(cogs) == len(datas) == len(logms) == len(index) == len(mr) > 30
    rel_out = np.abs(cogs[:, :, -1] - datas[:, :, -1]) / datas[:, :, -1]
    assert rel_out.max() < 1e-8, "pinned CoGs must match data at the outer radius"
    re = np.abs((cogs - datas) / datas).max(axis=2)
    assert np.allclose(re, mr, atol=1e-10), "stored max|rel| must match the curves"
    print(f"profile_qa.demo OK: n={len(cogs)}, pinned outer radius, mr consistent")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        sys.exit(main())
