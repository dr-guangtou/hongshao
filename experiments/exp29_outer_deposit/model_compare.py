"""exp29 — radial comparison of the four multi-epoch models.

The headline metric (max|rel| per epoch) hides *where* along the CoG each model fails.
This builds the full best-fit CoG at every radius for four models and compares them:
  1. independent (ceiling)  -- 5 free thetas, reconstructed from the cached per-epoch fits
  2. loose-zdep quad        -- each kernel param quadratic in observation z (15 params)
  3. puff: ratio law        -- one history, sigma *= (t_k/t_i)^q       (6 params)
  4. puff: diffusion law    -- one history, sigma^2 += kappa*(t_k-t_i) (6 params)

Outputs:
  - exp29_model_compare_cog   : example galaxies, CoG (data + 4 models), 5 epochs
  - exp29_model_compare_resid : same galaxies, relative residual (model-data)/data
  - exp29_model_compare_medprof: median relative-residual profile vs R, per epoch,
                                 in 3 stellar-mass bins

Fits are cached to outputs/model_compare.npz; pass --refit to recompute.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/model_compare.py [n] [--refit]
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
from run import dipfree_mah, ANCHOR_Z                                                # noqa: E402
from cog_extrapolate import measured_cog                                            # noqa: E402
from single_epoch_all import model_cog_epoch, anchor_times                          # noqa: E402
from loose_zdep import fit_joint as fit_loose                                       # noqa: E402
from puff_fit import fit as fit_puff                                                # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                           # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                        # noqa: E402

set_style()
FIGDIR = HERE / "figures"
NPZ_IN = HERE / "outputs" / "single_epoch_params.npz"
NPZ_OUT = HERE / "outputs" / "model_compare.npz"
R = COG_RAD_KPC
R_IN = 3.0
NAMES = ["independent", "loose-quad", "puff-ratio", "puff-diff"]
COLORS = {"independent": "0.45", "loose-quad": OKABE_ITO[1],
          "puff-ratio": OKABE_ITO[2], "puff-diff": OKABE_ITO[0]}


def models_for(gi, Pg, at):
    """Best-fit CoG (5 epochs x 24 radii) for the four models; None if any fit fails."""
    logC = measured_cog(gi)
    if logC is None:
        return None
    data = np.array([10.0 ** logC[k] for k in range(5)])          # (5,24)
    mah = dipfree_mah(gi); t_obs = mah["t_obs"]
    ind = []
    for k in range(5):                                            # reconstruct + aperture-pin
        m = model_cog_epoch(Pg[k], mah, k)
        ind.append(m * (data[k][-1] / m[-1]))
    loose = fit_loose(mah, data, t_obs, Pg, 2)
    ratio, _ = fit_puff(mah, data, t_obs, at, Pg, True, "ratio")
    diff, _ = fit_puff(mah, data, t_obs, at, Pg, True, "diff")
    if loose is None or ratio is None or diff is None:
        return None
    return data, np.array([np.array(ind), np.array(loose), np.array(ratio), np.array(diff)])


def compute(n):
    at = anchor_times()
    d = np.load(NPZ_IN)
    idx, P, logms = d["index"], d["params"], d["logms"]
    sel = np.linspace(0, len(idx) - 1, min(n, len(idx))).round().astype(int)
    gids, lms, datas, mods = [], [], [], []
    for i in sel:
        out = models_for(int(idx[i]), P[i], at)
        if out is None:
            continue
        datas.append(out[0]); mods.append(out[1]); gids.append(int(idx[i])); lms.append(logms[i])
    datas, mods = np.array(datas), np.array(mods)                 # (n,5,24), (n,4,5,24)
    np.savez(NPZ_OUT, index=np.array(gids), logms=np.array(lms), R=R, data=datas, models=mods)
    print(f"wrote {NPZ_OUT}  data {datas.shape}  models {mods.shape}")
    return np.load(NPZ_OUT)


def main():
    refit = "--refit" in sys.argv
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 60)
    d = compute(n) if (refit or not NPZ_OUT.exists()) else np.load(NPZ_OUT)
    logms, data, mods = d["logms"], d["data"], d["models"]
    resid = 100.0 * (mods - data[:, None, :, :]) / data[:, None, :, :]   # (n,4,5,24) %
    print(f"loaded n={len(logms)}; building figures")
    _fig_examples(logms, data, mods, kind="cog")
    _fig_examples(logms, data, mods, kind="resid", resid=resid)
    _fig_medprof(logms, resid)


def _examples(logms):
    o = np.argsort(logms)
    return [(o[-1], "most massive"), (o[len(o) // 2], "median mass"), (o[0], "least massive")]


def _fig_examples(logms, data, mods, kind, resid=None):
    rows = _examples(logms)
    fig, axes = plt.subplots(3, 5, figsize=(19, 10.5), sharex=True)
    inner = R <= R_IN
    for ri, (gi, lab) in enumerate(rows):
        for k, z in enumerate(ANCHOR_Z):
            ax = axes[ri, k]
            if kind == "cog":
                ax.plot(R, data[gi, k] / 1e11, "o", c="k", ms=3.2, zorder=5, label="TNG")
                for m, nm in enumerate(NAMES):
                    ax.plot(R, mods[gi, m, k] / 1e11, "-", c=COLORS[nm], lw=1.6, label=nm)
                ax.set(xscale="log", yscale="log")
            else:
                for m, nm in enumerate(NAMES):
                    ax.plot(R, resid[gi, m, k], "-", c=COLORS[nm], lw=1.5, label=nm)
                ax.axhline(0, c="0.6", lw=0.8); ax.set(xscale="log", ylim=(-12, 12))
            ax.axvspan(R.min(), R_IN, color="0.88", alpha=0.6)
            if ri == 0:
                ax.set_title(f"z = {z}")
            if k == 0:
                ax.set_ylabel(f"logM*={logms[gi]:.2f}\n" +
                              (r"$M_*(<R)\,[10^{11}M_\odot]$" if kind == "cog"
                               else "(model$-$data)/data [%]"), fontsize=9)
            if ri == 2:
                ax.set_xlabel("R [kpc]")
            if ri == 0 and k == 4:
                ax.legend(fontsize=7, loc="lower right" if kind == "cog" else "upper right")
    ttl = ("CoG fits (data dots, 4 models)" if kind == "cog"
           else "relative residuals of the 4 models")
    fig.suptitle(f"exp29 — {ttl}: example galaxies (rows) x epoch (cols); grey = inner 3 kpc",
                 fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / f"exp29_model_compare_{kind}")[0])


def _fig_medprof(logms, resid):
    order = np.argsort(logms); bins = np.array_split(order, 3)
    labels = ["low mass", "mid mass", "high mass"]
    fig, axes = plt.subplots(3, 5, figsize=(19, 10.5), sharex=True, sharey=True)
    for bi, (idx, lab) in enumerate(zip(bins, labels)):
        rng = f"logM*={logms[idx].min():.2f}-{logms[idx].max():.2f}  (n={len(idx)})"
        for k, z in enumerate(ANCHOR_Z):
            ax = axes[bi, k]
            for m, nm in enumerate(NAMES):
                med = np.median(resid[idx, m, k], axis=0)
                ax.plot(R, med, "-", c=COLORS[nm], lw=1.8, label=nm)
            ax.axhline(0, c="0.6", lw=0.8); ax.axvspan(R.min(), R_IN, color="0.88", alpha=0.6)
            ax.set(xscale="log", ylim=(-8, 8))
            if bi == 0:
                ax.set_title(f"z = {z}")
            if bi == 2:
                ax.set_xlabel("R [kpc]")
            if k == 0:
                ax.set_ylabel(f"{lab}\nmedian (model$-$data)/data [%]", fontsize=9)
            if bi == 0 and k == 4:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("exp29 — median relative-residual profile vs R, per epoch (cols) x stellar-mass "
                 "bin (rows); grey = inner 3 kpc", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "exp29_model_compare_medprof")[0])


def demo():
    """Self-check: reconstructed models are aperture-pinned to the data."""
    at = anchor_times()
    d = np.load(NPZ_IN)
    out = models_for(int(d["index"][0]), d["params"][0], at)
    assert out is not None
    data, mods = out
    for m in range(4):
        for k in range(5):
            assert abs(mods[m, k, -1] / data[k, -1] - 1.0) < 1e-6, (m, k)
    print("model_compare.demo OK: all 4 models aperture-pinned at 148 kpc")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        sys.exit(main())
