"""exp29 — does the SINGLE-epoch (z=0.4) CoG fit predict higher-z CoGs?

exp25 fit the deposition kernel (centred Gaussian + two-epoch efficiency +
sigma(t)) to one galaxy's z=0.4 curve of growth and reached ~0.008 dex. Because
the kernel builds the galaxy epoch-by-epoch, those SAME fitted parameters already
predict the galaxy at every earlier epoch — integrate the deposits only to t(z_k),
no refitting. This is the cleanest version of the "stop at each redshift" test and
isolates the question the multi-epoch fit raised: is it the *joint* fit that is
hard, or does the single-epoch model simply fail to generalize across cosmic time?

Per galaxy: fit (sigma_0, g, b_early, b_late, z_c) to the **z=0.4 CoG only** (the
measured `load_aper` curve of growth, 24 radii, dip-free DiffMAH MAH), then predict
the CoG at z = 0.7/1.0/1.5/2.0 with the frozen parameters and compare to the
measured CoG. We separate the prediction error into the total-mass growth (the
efficiency's job) and the profile shape at fixed total (the sigma(t) time->radius
rule). No free Sersic — the deposit stays a centred Gaussian (deposit_cog, p=0).

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/cog_extrapolate.py
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
from profile_fit import pick_galaxies                                               # noqa: E402
from hongshao.tng_data import load_aper, COG_RAD_KPC, ANCHORS                        # noqa: E402
from hongshao.plotting import set_style, save_fig                                   # noqa: E402

set_style()
FIGDIR = HERE / "figures"
R = COG_RAD_KPC
ZKEYS = [ANCHORS[z][0] for z in ANCHOR_Z]            # z0p4..z2 in epoch order


def measured_cog(gi):
    """log10 measured CoG at the 5 epochs (24 radii each); None if any missing."""
    a = load_aper(gi)
    out = []
    for zk in ZKEYS:
        c = np.asarray(a[zk]["cog"], float)
        if c.shape != (24,) or not np.isfinite(c).all() or (c <= 0).any():
            return None
        out.append(np.log10(c))
    return np.array(out)                             # (5, 24)


def fit_z0p4_predict(gi, logC):
    """Fit Gaussian kernel to z=0.4 CoG, return predicted log CoG at all 5 epochs."""
    mah = dipfree_mah(gi)
    Mstar_tot = 10.0 ** logC[0, -1]                  # measured z=0.4 total pins normalization

    def model_epochs(q):
        s0, g, be, bl, zc = 10.0 ** q[0], q[1], q[2], q[3], q[4]
        dMstar = deposited(eff_two_epoch(mah["z"], be, bl, zc), mah["dMh"], Mstar_tot)
        sigma = width_t(s0, g, mah["t"], mah["t_obs"])
        return [deposit_cog(dMstar[mah["snap"] <= sa], sigma[mah["snap"] <= sa], 0.0, R)
                for sa in ANCHOR_SNAP]

    def loss(q):                                     # z=0.4 CoG only
        m0 = np.log10(np.clip(model_epochs(q)[0], 1.0, None))
        return float(np.sqrt(np.mean((m0 - logC[0]) ** 2)))

    best = None
    for zc0 in (1.0, 2.0, 3.0, 4.0):
        res = minimize(loss, [np.log10(40.0), 1.5, 4.0, 1.5, zc0], method="Nelder-Mead",
                       options=dict(maxiter=8000, xatol=1e-4, fatol=1e-9))
        if best is None or res.fun < best.fun:
            best = res
    pred = np.array([np.log10(np.clip(c, 1.0, None)) for c in model_epochs(best.x)])
    return pred, best.fun


def main():
    gxs = pick_galaxies()
    print("exp29 — fit z=0.4 CoG, predict higher-z CoG (frozen params, dip-free MAH):\n")
    print(f"  {'idx':>5s} {'logM*':>6s} | {'z=0.4(fit)':>10s} " +
          " ".join(f"z={z}".rjust(8) for z in ANCHOR_Z[1:]) + "   | total-mass dlog @z=2")
    results = []
    for logm, gi in gxs:
        logC = measured_cog(gi)
        if logC is None:
            continue
        pred, rms0 = fit_z0p4_predict(gi, logC)
        rms = [float(np.sqrt(np.mean((pred[k] - logC[k]) ** 2))) for k in range(5)]
        dmass_z2 = pred[4, -1] - logC[4, -1]         # predicted vs measured total at z=2
        results.append((logm, gi, logC, pred, rms))
        print(f"  {gi:5d} {logm:6.2f} | {rms[0]:10.3f} " +
              " ".join(f"{rms[k]:8.3f}" for k in range(1, 5)) + f"   | {dmass_z2:+.2f}")
    med = np.median([r[4] for r in results], axis=0)
    print(f"\n  median CoG RMS per epoch: z=0.4 {med[0]:.3f} (fit) -> "
          f"z=0.7 {med[1]:.3f}, z=1.0 {med[2]:.3f}, z=1.5 {med[3]:.3f}, z=2.0 {med[4]:.3f} dex")
    _figure(results)


def _figure(results):
    sel = [results[0], results[len(results) // 2], results[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    cmap = matplotlib.colormaps["viridis"]
    for ax, (logm, gi, logC, pred, rms) in zip(axes, sel):
        for k, z in enumerate(ANCHOR_Z):
            c = cmap(k / 4)
            ax.plot(R, logC[k], "o", c=c, ms=3, alpha=0.75)
            ls = "-" if k == 0 else "--"             # solid = fitted z=0.4, dashed = predicted
            ax.plot(R, pred[k], ls, c=c, lw=1.8, label=f"z={z} ({rms[k]:.2f})")
        ax.set(xscale="log", xlabel="R [kpc]", ylabel=r"$\log M_*(<R)\ [M_\odot]$",
               title=f"logM*={logm:.2f} (idx {gi})")
        ax.legend(fontsize=7, title="epoch (RMS dex)", loc="lower right")
    fig.suptitle("exp29 — fit z=0.4 CoG (solid), predict higher-z CoG (dashed) vs TNG (dots); "
                 "centred Gaussian, frozen params", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp29_cog_extrapolate")[0])


if __name__ == "__main__":
    sys.exit(main())
