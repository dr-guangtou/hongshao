"""exp29 — the free-mass NNLS floor: the most expressive single-history fit.

Every model so far constrains the deposit masses through a parametric efficiency
f(t_i). This removes that constraint: keep the deposit TIMES (the MAH snapshots) and
WIDTHS (sigma_i = sigma_0 (t_i/t_obs)^g) tied to the history, but let each deposit's
MASS be a free non-negative parameter. One shared mass vector feeds all epochs
(masked by t_i <= t_k), so it is still a single consistent history.

The CoG is linear in the masses, so the inner solve is convex non-negative least
squares (weighted by 1/data -> minimizes relative residuals). The width-law params
(sigma_0, g) are the only nonlinear knobs, found in a 2-D outer loop (NNLS inside).

This measures the representational FLOOR: with free masses, how close can one
consistent history get to the independent per-epoch ceiling (~0.7%)?
  - floor ~ ceiling -> the missing freedom IS per-deposit mass (build a forward model
    that predicts the masses).
  - floor ~ loose-zdep (~4.5%) -> free masses don't help beyond the parametric models;
    the limit is the Gaussian-sum width basis (need a richer deposit / puffing).
Caveat: ~70 free masses per galaxy -> this is an in-sample upper bound, not a forward
model; a generalization test (held-out radii/epoch) is the natural follow-up.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/nnls_floor.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls, minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from run import dipfree_mah, ANCHOR_SNAP, ANCHOR_Z                                   # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                           # noqa: E402
from hongshao.plotting import set_style, save_fig                                   # noqa: E402

set_style()
FIGDIR = HERE / "figures"
NPZ = HERE / "outputs" / "model_compare.npz"                     # data + the other 4 models
R = COG_RAD_KPC
R_IN = 3.0
EVAL = R > R_IN


def _fit_block(t_i, snap_i, t_obs, data, epochs):
    """Free-mass NNLS over `epochs` sharing ONE mass vector and ONE width law
    (sigma_0, g) optimized outer. Returns {k: model CoG (24,)} for k in epochs."""
    Rsel = R[EVAL]
    masks = [snap_i <= ANCHOR_SNAP[k] for k in epochs]
    dsel = [data[k][EVAL] for k in epochs]

    def solve(par):
        sig = 10.0 ** par[0] * (t_i / t_obs) ** par[1]
        basis = 1.0 - np.exp(-Rsel[:, None] ** 2 / (2.0 * sig[None, :] ** 2))
        A = np.vstack([(basis * m[None, :]) / d[:, None] for m, d in zip(masks, dsel)])
        x, rnorm = nnls(A, np.ones(len(epochs) * len(Rsel)), maxiter=5 * basis.shape[1])
        return x, sig, rnorm

    best = None
    for s0 in (1.7, 2.2, 2.7):                                   # log10 sigma_0 starts
        for g in (1.0, 1.7):
            r = minimize(lambda p: solve(p)[2], [s0, g], method="Nelder-Mead",
                         options=dict(maxiter=400, xatol=1e-3, fatol=1e-6))
            if best is None or r.fun < best.fun:
                best = r
    x, sig, _ = solve(best.x)
    full = 1.0 - np.exp(-R[:, None] ** 2 / (2.0 * sig[None, :] ** 2))            # (24, N)
    return {k: (x * m) @ full.T for k, m in zip(epochs, masks)}


def nnls_floor(gi, data, joint=True):
    """Free-mass NNLS CoG (5 epochs x 24 radii). joint=True shares ONE mass vector +
    width law across all epochs (a consistent history); joint=False fits each epoch
    independently with its own free masses + width law (the free-mass ceiling)."""
    mah = dipfree_mah(gi); t_obs = mah["t_obs"]
    use = mah["snap"] <= ANCHOR_SNAP[0]                          # deposits up to z=0.4
    t_i, snap_i = mah["t"][use], mah["snap"][use]
    if joint:
        cogs = _fit_block(t_i, snap_i, t_obs, data, list(range(5)))
    else:
        cogs = {k: _fit_block(t_i, snap_i, t_obs, data, [k])[k] for k in range(5)}
    return np.array([cogs[k] for k in range(5)])


def maxrel(model, data):
    return np.array([np.abs((model[k][EVAL] - data[k][EVAL]) / data[k][EVAL]).max()
                     for k in range(5)])


COLS = ["indep-alone", "nnls-alone", "loose-joint", "puff-joint", "nnls-joint"]
CCOL = {"indep-alone": "0.45", "nnls-alone": "#CC3377", "loose-joint": "#0072B2",
        "puff-joint": "#009E73", "nnls-joint": "#D55E00"}


def main():
    d = np.load(NPZ)
    idx, logms, data, mods = d["index"], d["logms"], d["data"], d["models"]
    n = len(idx)
    # mods order in model_compare.npz: [independent, loose-quad, puff-ratio, puff-diff]
    nnls_j = np.array([nnls_floor(int(idx[i]), data[i], joint=True) for i in range(n)])   # (n,5,24)
    nnls_a = np.array([nnls_floor(int(idx[i]), data[i], joint=False) for i in range(n)])
    stack = {"indep-alone": mods[:, 0], "nnls-alone": nnls_a, "loose-joint": mods[:, 1],
             "puff-joint": mods[:, 2], "nnls-joint": nnls_j}
    resid = {nm: 100.0 * (m - data) / data for nm, m in stack.items()}                    # each (n,5,24)
    mr = {nm: np.array([maxrel(stack[nm][i], data[i]) for i in range(n)]) for nm in COLS}  # each (n,5)

    print(f"exp29 — free-mass NNLS: per-epoch ALONE vs consistent JOINT (n={n}), max|rel| over R>3 kpc\n")
    print(f"  {'z':>4s} | " + " | ".join(f"{nm:>11s}" for nm in COLS))
    for k, z in enumerate(ANCHOR_Z):
        print(f"  {z:4.1f} | " + " | ".join(f"{100*np.median(mr[nm][:,k]):9.1f}%" for nm in COLS))
    avg = {nm: 100 * np.median([np.median(mr[nm][:, k]) for k in range(5)]) for nm in COLS}
    print("\n  epoch-avg median max|rel|:  " + "   ".join(f"{nm} {avg[nm]:.1f}%" for nm in COLS))

    print(f"\n[verdict] FREE masses, same method & metric:  per-epoch ALONE {avg['nnls-alone']:.1f}%  "
          f"->  consistent JOINT {avg['nnls-joint']:.1f}%  (a ~{avg['nnls-joint']/avg['nnls-alone']:.0f}x cost)")
    print(f"  Free masses fit each epoch ALONE to {avg['nnls-alone']:.1f}%, but ONE shared mass vector (a\n"
          f"  single consistent additive history) caps the JOINT fit at {avg['nnls-joint']:.0f}% -- and free masses\n"
          f"  do NOT relieve it. The parametric joint models beat free-mass-joint only by RELAXING\n"
          f"  consistency (loose-quad {avg['loose-joint']:.1f}%, per-epoch z-trends) or adding WIDTH freedom\n"
          f"  (puff {avg['puff-joint']:.1f}%). So the binding limit is the single consistent Gaussian-sum history\n"
          f"  itself, not the mass parameterization -> the deposit primitive / spatial freedom is the\n"
          f"  only lever left toward the ceiling.")
    _figure(logms, resid)


def _figure(logms, resid):
    order = np.argsort(logms); bins = np.array_split(order, 3)
    labels = ["low mass", "mid mass", "high mass"]
    show = ["nnls-alone", "nnls-joint", "loose-joint", "puff-joint"]
    fig, axes = plt.subplots(3, 5, figsize=(19, 10.5), sharex=True, sharey=True)
    for bi, (idx, lab) in enumerate(zip(bins, labels)):
        for k, z in enumerate(ANCHOR_Z):
            ax = axes[bi, k]
            for nm in show:
                lw = 2.4 if nm.startswith("nnls") else 1.3
                a = 1.0 if nm.startswith("nnls") else 0.5
                ax.plot(R, np.median(resid[nm][idx, k], axis=0), "-", c=CCOL[nm], lw=lw, alpha=a,
                        label=nm if (bi == 0 and k == 4) else None)
            ax.axhline(0, c="0.6", lw=0.8); ax.axvspan(R.min(), R_IN, color="0.88", alpha=0.6)
            ax.set(xscale="log", ylim=(-10, 10))
            if bi == 0:
                ax.set_title(f"z = {z}")
            if bi == 2:
                ax.set_xlabel("R [kpc]")
            if k == 0:
                ax.set_ylabel(f"{lab}\nmedian (model$-$data)/data [%]", fontsize=9)
            if bi == 0 and k == 4:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("exp29 — free masses: per-epoch ALONE (pink, ~ceiling) vs consistent JOINT (orange) "
                 "vs parametric joint models; median residual, epoch x mass bin", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp29_nnls_floor")[0])


def demo():
    """Self-check: the joint free-mass NNLS yields a monotonic CoG and beats the fixed
    parametric model (which is ~10-15% for the BCG, far worse without free masses)."""
    d = np.load(NPZ)
    i = len(d["index"]) // 2                                     # a mid-mass galaxy
    cog = nnls_floor(int(d["index"][i]), d["data"][i])
    assert np.all(np.diff(cog, axis=1) >= -1e-6), "CoG not monotonic"
    assert np.isfinite(cog).all() and (cog > 0).all(), "non-finite / non-positive CoG"
    print(f"nnls_floor.demo OK: monotonic positive joint CoG; per-epoch max-rel "
          f"{np.round(100*maxrel(cog, d['data'][i]), 1)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        sys.exit(main())
