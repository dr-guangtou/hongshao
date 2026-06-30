"""exp29 — single-epoch vs multi-epoch fit, judged on the z=0.4 CoG.

Verify the "single-epoch fit is great" impression and quantify what multi-epoch
costs at z=0.4. SAME model (two-epoch fraction + power-law width + centred Gaussian)
fit two ways, per galaxy:
  single = fit the z=0.4 CoG only (the original exp25 / cog_extrapolate setup)
  multi  = fit all 5 epoch CoGs jointly
Then compare the z=0.4 CoG: data vs both models, in LINEAR M* and relative residual
(model-data)/data, and report the z=0.4 log-RMS and the worst relative error.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/single_vs_multi.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from deposit import deposit_cog, width_t, eff_two_epoch, deposited                  # noqa: E402
from run import dipfree_mah, ANCHOR_SNAP, ANCHOR_Z                                   # noqa: E402
from cog_extrapolate import measured_cog                                            # noqa: E402
from profile_fit import pick_galaxies                                               # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                           # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                        # noqa: E402

set_style()
FIGDIR = HERE / "figures"
R = COG_RAD_KPC


def model_epochs(q, mah, Mstar_tot):
    """Two-epoch fraction + power-law width + centred Gaussian -> CoG at 5 epochs."""
    s0, g, be, bl, zc = 10.0 ** q[0], q[1], q[2], q[3], q[4]
    dMstar = deposited(eff_two_epoch(mah["z"], be, bl, zc), mah["dMh"], Mstar_tot)
    sigma = width_t(s0, g, mah["t"], mah["t_obs"])
    return [deposit_cog(dMstar[mah["snap"] <= sa], sigma[mah["snap"] <= sa], 0.0, R)
            for sa in ANCHOR_SNAP]


def _rms(model_cog, logC_k):
    return float(np.sqrt(np.mean((np.log10(np.clip(model_cog, 1.0, None)) - logC_k) ** 2)))


def fit(gi, logC, multi):
    """Fit the model to z=0.4 only (multi=False) or all 5 epochs (multi=True)."""
    mah = dipfree_mah(gi); Mstar_tot = 10.0 ** logC[0, -1]

    def loss(q):
        m = model_epochs(q, mah, Mstar_tot)
        return float(np.mean([_rms(m[k], logC[k]) for k in range(5)])) if multi else _rms(m[0], logC[0])

    best = None
    for zc0 in (1.0, 2.0, 3.0, 4.0):
        r = minimize(loss, [np.log10(40.0), 1.5, 4.0, 1.5, zc0], method="Nelder-Mead",
                     options=dict(maxiter=8000, xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    return np.array(model_epochs(best.x, mah, Mstar_tot))


def main():
    # stats over a representative sample
    gxs = pick_galaxies(40)
    rows = []
    for logm, gi in gxs:
        logC = measured_cog(gi)
        if logC is None:
            continue
        s = fit(gi, logC, multi=False)[0]; m = fit(gi, logC, multi=True)[0]
        data = 10.0 ** logC[0]
        rows.append((_rms(s, logC[0]), _rms(m, logC[0]),
                     np.abs((s - data) / data).max(), np.abs((m - data) / data).max()))
    rows = np.array(rows)
    print("z=0.4 CoG fit quality (n=%d): single-epoch vs multi-epoch, SAME model\n" % len(rows))
    print(f"  log-RMS   : single median {np.median(rows[:,0]):.3f}  |  multi median {np.median(rows[:,1]):.3f}")
    print(f"  max|rel|  : single median {np.median(rows[:,2]):.3f}  |  multi median {np.median(rows[:,3]):.3f}")
    print(f"  (single-epoch z=0.4 is {'much ' if np.median(rows[:,1])>3*np.median(rows[:,0]) else ''}"
          f"better at z=0.4, as expected)")

    # figure: 5 example galaxies, z=0.4 CoG linear + relative residual
    show = pick_galaxies(5)
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.4), sharex=True)
    for col, (logm, gi) in enumerate(show):
        logC = measured_cog(gi)
        if logC is None:
            continue
        s = fit(gi, logC, multi=False)[0]; m = fit(gi, logC, multi=True)[0]; data = 10.0 ** logC[0]
        aC, aR = axes[0, col], axes[1, col]
        aC.plot(R, data / 1e11, "o", c="k", ms=4, label="TNG z=0.4")
        aC.plot(R, s / 1e11, "-", c=OKABE_ITO[1], lw=1.8, label=f"single ({_rms(s,logC[0]):.3f})")
        aC.plot(R, m / 1e11, "--", c=OKABE_ITO[5], lw=1.8, label=f"multi ({_rms(m,logC[0]):.3f})")
        aR.plot(R, 100 * (s - data) / data, "-", c=OKABE_ITO[1], lw=1.6)
        aR.plot(R, 100 * (m - data) / data, "--", c=OKABE_ITO[5], lw=1.6)
        aR.axhline(0, c="0.6", lw=0.8); aR.axvspan(R.min(), 3, color="0.85", alpha=0.5)
        aC.set(xscale="log", title=f"logM*={logm:.2f}"); aR.set(xscale="log", xlabel="R [kpc]", ylim=(-20, 20))
        aC.legend(fontsize=7, loc="upper left")
        if col == 0:
            aC.set_ylabel(r"$M_*(<R)\ [10^{11}M_\odot]$"); aR.set_ylabel(r"(model$-$data)/data [%]")
    fig.suptitle("exp29 — z=0.4 CoG fit: single-epoch (solid) vs multi-epoch (dashed) vs TNG (dots); "
                 "linear M* (top), relative residual (bottom)", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp29_single_vs_multi_z0p4")[0])


if __name__ == "__main__":
    sys.exit(main())
