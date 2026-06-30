"""exp29 — fit the FULL multi-epoch surface-density profile (not the slope).

The b(z) result in run.py matched only the *tilt* of the inter-epoch growth; the
profiles themselves are off in amount (~0.36 dex, up to 0.7 dex on the long
baseline) because run.py froze the level (efficiency, sigma_0) and fit one
parameter to the slope. Here we fit each galaxy's whole Sigma(R) at all 5 epochs,
freeing the temporal budget (two-epoch efficiency) and the spatial kernel
(sigma_0, g, p), so the model can match the *amount* of growth, not just its tilt.

Target: log Sigma(R) at z=0.4/0.7/1.0/1.5/2.0, over R in [R_IN, 100] kpc (the
inner ~3 kpc is downweighted — TNG softening), with ONE global per-galaxy offset
marginalized (the ~constant ellipticity overshoot of the isophote Sigma; the
*relative* level between epochs stays as signal, so the efficiency is tested).

Validate on a handful of galaxies spanning the mass range, p=0 (centred Gaussian)
vs p-free, before any population run.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/profile_fit.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from deposit import deposit_sigma_sersic, width_t, eff_two_epoch, deposited          # noqa: E402
from run import dipfree_mah, ANCHOR_SNAP, ANCHOR_Z, OFFICIAL, DIFFPROF, TABLE        # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR = HERE / "figures"
R_IN, R_OUT = 3.0, 100.0                         # downweight inner ~3 kpc (TNG softening)


def fit_profiles(gi, R, Sig, flags, Mstar_tot, n_free):
    """Fit (a_0, g, b_early, b_late, z_c[, n]) to the 5-epoch log Sigma(R).

    Deposit = Sersic of index n (n=0.5 fixed -> the exp25/exp29 centred Gaussian)."""
    mah = dipfree_mah(gi)
    fitmask = [flags[k] & (R >= R_IN) & (R <= R_OUT) & (Sig[k] > 0) for k in range(5)]

    def model(q):
        a0, g, be, bl, zc = 10.0 ** q[0], float(np.clip(q[1], 0.0, 5.0)), q[2], q[3], q[4]
        n = float(np.clip(q[5], 0.3, 8.0)) if n_free else 0.5
        dMstar = deposited(eff_two_epoch(mah["z"], be, bl, zc), mah["dMh"], Mstar_tot)
        scale = width_t(a0, g, mah["t"], mah["t_obs"])
        return {sa: deposit_sigma_sersic(dMstar[mah["snap"] <= sa], scale[mah["snap"] <= sa], n, R)
                for sa in ANCHOR_SNAP}

    def resid(q):
        Sm = model(q)
        d = [np.log10(np.clip(Sm[ANCHOR_SNAP[k]][fitmask[k]], 1e-3, None)) - np.log10(Sig[k][fitmask[k]])
             for k in range(5) if fitmask[k].sum()]
        r = np.concatenate(d)
        return r - r.mean()                       # one global offset (ellipticity)

    best = None
    n_starts = (0.5, 2.0, 4.0) if n_free else (None,)
    for zc0 in (1.0, 2.0, 3.0):
        for g0 in (0.8, 2.5):
            for n0 in n_starts:
                q0 = [np.log10(40.0), g0, 3.0, 1.0, zc0] + ([n0] if n_free else [])
                res = minimize(lambda q: np.sqrt(np.mean(resid(q) ** 2)), q0, method="Nelder-Mead",
                               options=dict(maxiter=12000, xatol=1e-4, fatol=1e-8))
                if best is None or res.fun < best.fun:
                    best = res
    q = best.x
    par = dict(a0=10.0 ** q[0], g=float(np.clip(q[1], 0.0, 5.0)), b_early=q[2], b_late=q[3], z_c=q[4],
               n=(float(np.clip(q[5], 0.3, 8.0)) if n_free else 0.5))
    return dict(rms=float(best.fun), par=par, model=model(q), fitmask=fitmask)


def pick_galaxies(n=6):
    """BCG + a spread across the M* range (use-sample with valid 5-epoch profiles)."""
    t = Table.read(TABLE); cog = np.asarray(t["logmstar_cog"], float)
    idx = np.asarray(t["index"]); use = np.asarray(t["use"])
    ok = np.isfinite(cog).all(1) & use
    dp = np.load(DIFFPROF); have = {int(g) for g in dp["index"]}
    mz = np.load(OFFICIAL); hmz = {int(g) for g, ok_ in zip(mz["index"], mz["matched"]) if ok_}
    cand = [(float(cog[i, -1]), int(idx[i])) for i in np.where(ok)[0]
            if int(idx[i]) in have and int(idx[i]) in hmz]
    cand.sort(reverse=True)
    pct = np.linspace(0, len(cand) - 1, n).round().astype(int)        # top + evenly down
    return [cand[j] for j in pct]


def main():
    R = np.load(DIFFPROF)["R"]; dp = np.load(DIFFPROF)
    dpm = {int(g): r for r, g in enumerate(dp["index"])}
    t = Table.read(TABLE); cogmap = {int(g): 10.0 ** float(c[-1])
                                     for g, c in zip(np.asarray(t["index"]),
                                                     np.asarray(t["logmstar_cog"], float)) if np.isfinite(c[-1])}
    gxs = pick_galaxies()
    print(f"exp29 profile fit — full Sigma(R), R in [{R_IN},{R_OUT}] kpc, one global offset:")
    print("  deposit = Sersic index n  (n=0.5 == the centred Gaussian)\n")
    print(f"  {'logM*':>6s} {'index':>6s} {'Gauss(n=.5)':>11s} {'Sersic(n free)':>14s} "
          f"{'n':>5s} {'g':>5s} {'z_c':>5s}")
    results = []
    for logm, gi in gxs:
        Sig, flags = dp["sigma"][dpm[gi]], dp["flags"][dpm[gi]]
        f0 = fit_profiles(gi, R, Sig, flags, cogmap[gi], False)
        fp = fit_profiles(gi, R, Sig, flags, cogmap[gi], True)
        results.append((logm, gi, Sig, flags, f0, fp))
        print(f"  {logm:6.2f} {gi:6d} {f0['rms']:11.3f} {fp['rms']:14.3f} "
              f"{fp['par']['n']:5.2f} {fp['par']['g']:5.2f} {fp['par']['z_c']:5.2f}")
    med0 = np.median([r[4]["rms"] for r in results])
    medp = np.median([r[5]["rms"] for r in results])
    print(f"\n  median full-profile RMS: Gaussian {med0:.3f} dex   Sersic-n {medp:.3f} dex   "
          f"(gain {med0 - medp:+.3f}); median best n={np.median([r[5]['par']['n'] for r in results]):.2f}")
    _figure(R, results)


def _figure(R, results):
    sel = [results[0], results[len(results) // 2], results[-1]]   # BCG, mid, low-mass
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    cmap = matplotlib.colormaps["viridis"]
    for ax, (logm, gi, Sig, flags, f0, fp) in zip(axes, sel):
        for k, z in enumerate(ANCHOR_Z):
            c = cmap(k / 4); g = flags[k] & (Sig[k] > 0)
            ax.plot(R[g], np.log10(Sig[k][g]), "o", c=c, ms=3, alpha=0.7)
            ax.plot(R, np.log10(np.clip(fp["model"][ANCHOR_SNAP[k]], 1e-3, None)), "-", c=c, lw=1.7,
                    label=f"z={z}" if k % 2 == 0 else None)
        ax.axvspan(R.min(), R_IN, color="0.85", alpha=0.5)        # downweighted inner region
        ax.set(xscale="log", xlabel="R [kpc]", ylabel=r"$\log\Sigma_*$ [$M_\odot/{\rm kpc}^2$]",
               title=f"logM*={logm:.2f} (idx {gi})  n={fp['par']['n']:.2f}  "
                     f"RMS {f0['rms']:.2f}(Gauss)$\\to${fp['rms']:.2f}")
        ax.legend(fontsize=7, title="epoch", loc="upper right")
    fig.suptitle("exp29 — full multi-epoch Sigma(R): Sersic-n deposit (lines) vs TNG (dots); "
                 "grey = inner 3 kpc downweighted", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp29_profile_fit")[0])


if __name__ == "__main__":
    sys.exit(main())
